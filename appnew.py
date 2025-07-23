import streamlit as st
import streamlit.components.v1 as components
import codecs
import pandas as pd
import numpy as np
import os
import random
import pandas as pd
import holidays
from faker import Faker
from collections import namedtuple, OrderedDict, defaultdict
import ast 
import numpy as np
import datetime as dt
import re
import json 
import sys
import time
import copy
import io
sys.path.append('..')

# 遺伝的アルゴリズム+局所探索クラス
class GeneticScheduler:
    def __init__(self, n_staff, n_day, job, requirement, day_off, avoid_jobs, 
                 early, late, num_off, B, LB, weights):
        self.n_staff = n_staff
        self.n_day = n_day
        self.job = job
        self.requirement = requirement
        self.day_off = day_off
        self.avoid_jobs = avoid_jobs
        self.early = early
        self.late = late
        self.num_off = num_off
        self.B = B
        self.LB = LB
        self.weights = weights
        
        # 遺伝的アルゴリズムのパラメータ
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
    def create_individual(self):
        """個体（勤務シフト案）を生成する"""
        individual = np.zeros((self.n_staff, self.n_day), dtype=int)
        
        for i in range(self.n_staff):
            # 先に休日を設定
            rest_days = random.sample(range(self.n_day), self.B[i])
            for day in rest_days:
                individual[i][day] = 0
            
            # 勤務日に仕事を割り当てる
            work_days = [d for d in range(self.n_day) if d not in rest_days]
            available_jobs = [j for j in self.job if j != 0 and j not in self.avoid_jobs[i]]
            
            for day in work_days:
                if day in self.day_off[i]:
                    individual[i][day] = 0  # 休み希望日は休みに設定
                else:
                    individual[i][day] = random.choice(available_jobs) if available_jobs else 0
        
        return individual
    
    def calculate_fitness(self, individual):
        """個体の適応度を計算する（制約違反のペナルティスコア、低いほど良い）"""
        penalty = 0
        
        # 1. 休み希望日出勤制約
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if t in self.day_off[i] and individual[i][t] != 0:
                    penalty += self.weights['obj_weight']
        
        # 2. 5日連続勤務制約
        for i in range(self.n_staff):
            for t in range(self.n_day - 5):
                consecutive_work = sum(1 for s in range(t, t + 6) if individual[i][s] != 0)
                if consecutive_work > 5:
                    penalty += self.weights['UB_max5_weight'] * (consecutive_work - 5)
        
        # 3. 4日連続勤務制約
        for i in range(self.n_staff):
            for t in range(self.n_day - 4):
                consecutive_work = sum(1 for s in range(t, t + 5) if individual[i][s] != 0)
                if consecutive_work > 4:
                    penalty += self.weights['UB_max4_weight'] * (consecutive_work - 4)
        
        # 4. 4日連続休み制約
        for i in range(self.n_staff):
            for t in range(self.n_day - 3):
                if t + 4 <= self.n_day:
                    consecutive_days = set(range(t, t + 4)) - self.day_off[i]
                    work_in_period = sum(1 for s in consecutive_days if s < self.n_day and individual[i][s] != 0)
                    if work_in_period == 0:
                        penalty += self.weights['LB_min1_weight']
        
        # 5. 当日必要人数下限制約
        for t in range(self.n_day):
            for j in self.job:
                if j != 0:
                    actual_count = sum(1 for i in range(self.n_staff) if individual[i][t] == j)
                    if actual_count < self.LB[t, j]:
                        penalty += self.weights['LBC_weight'] * (self.LB[t, j] - actual_count)
        
        # 6. 担当不可能な勤務制約
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if individual[i][t] in self.avoid_jobs[i]:
                    penalty += 1000  # 重大な違反
        
        # 7. Staff1とStaff2の選択的勤務
        for t in range(self.n_day):
            if individual[1][t] == 0 and individual[2][t] == 0:
                penalty += self.weights['Disjective_weight']
        
        # 8. 休-勤務-休パターンの回避
        for i in range(self.n_staff):
            for t in range(self.n_day - 2):
                if (individual[i][t] == 0 and individual[i][t+1] != 0 and individual[i][t+2] == 0):
                    penalty += self.weights['RestWorkRest_weight']
        
        # 9. 遅番・早番の連続回避
        for i in range(self.n_staff):
            for t in range(self.n_day - 1):
                if (individual[i][t] in self.early and individual[i][t+1] in self.late):
                    penalty += self.weights['LateEarly_weight']
        
        # 10. 月間休日数制約
        for i in range(self.n_staff):
            rest_days = sum(1 for t in range(self.n_day) if individual[i][t] == 0)
            if rest_days != self.B[i]:
                penalty += self.weights['num_off_weight'] * abs(rest_days - self.B[i])
        
        return -penalty  # 適応度を最大化するため、負の値にする
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # ランダムに交叉点を選択
        cross_point = random.randint(0, self.n_day - 1)
        
        for i in range(self.n_staff):
            # 交叉点以降の遺伝子を交換
            child1[i][cross_point:] = parent2[i][cross_point:]
            child2[i][cross_point:] = parent1[i][cross_point:]
        
        return child1, child2
    
    def mutate(self, individual):
        """突然変異操作"""
        mutated = individual.copy()
        
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if random.random() < self.mutation_rate:
                    available_jobs = [j for j in self.job if j not in self.avoid_jobs[i]]
                    mutated[i][t] = random.choice(available_jobs)
        
        return mutated
    
    def local_search(self, individual):
        """局所探索による最適化"""
        best = individual.copy()
        best_fitness = self.calculate_fitness(best)
        
        # 各スタッフのシフトを改善しようと試みる
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if t not in self.day_off[i]:  # 休み希望日ではない
                    available_jobs = [j for j in self.job if j not in self.avoid_jobs[i]]
                    current_job = individual[i][t]
                    
                    for job in available_jobs:
                        if job != current_job:
                            # 新しい勤務割り当てを試す
                            test_individual = individual.copy()
                            test_individual[i][t] = job
                            fitness = self.calculate_fitness(test_individual)
                            
                            if fitness > best_fitness:
                                best = test_individual.copy()
                                best_fitness = fitness
        
        return best
    
    def solve(self, time_limit=30, progress_callback=None):
        """遺伝的アルゴリズム+局所探索で求解"""
        start_time = time.time()  # 開始時間を記録
        
        # 初期集団を生成
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = float('-inf')
        
        generation = 0
        # 重要：whileループの条件で時間制限をチェック
        while generation < self.generations and (time.time() - start_time) < time_limit:
            # 進捗コールバックを更新
            if progress_callback:
                progress = min(generation / self.generations, (time.time() - start_time) / time_limit)
                progress_callback(progress, generation, best_fitness)
            
            # 適応度を計算
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 最良個体を更新
            if fitness_scores[0][1] > best_fitness:
                best_individual = fitness_scores[0][0].copy()
                best_fitness = fitness_scores[0][1]
            
            # エリートを選択
            elite = [ind for ind, _ in fitness_scores[:self.elite_size]]
            
            # 新世代を生成
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # トーナメント選択
                tournament_size = 3
                parents = random.sample(fitness_scores[:self.population_size//2], 2)
                parent1, parent2 = parents[0][0], parents[1][0]
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 突然変異
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            generation += 1
            
            # 10世代ごとに局所探索を実行
            if generation % 10 == 0 and best_individual is not None:
                improved = self.local_search(best_individual)
                improved_fitness = self.calculate_fitness(improved)
                if improved_fitness > best_fitness:
                    best_individual = improved
                    best_fitness = improved_fitness
        
        return best_individual, best_fitness, generation

def create_random_schedule(n_staff, n_day, job, avoid_jobs, B):
    """初期表示用にランダムな勤務シフト表を生成する"""
    schedule = np.zeros((n_staff, n_day), dtype=int)
    
    for i in range(n_staff):
        # 先に休日を設定
        rest_days = random.sample(range(n_day), B[i])
        for day in rest_days:
            schedule[i][day] = 0
        
        # 勤務日に仕事を割り当てる
        work_days = [d for d in range(n_day) if d not in rest_days]
        available_jobs = [j for j in job if j != 0 and j not in avoid_jobs[i]]
        
        for day in work_days:
            schedule[i][day] = random.choice(available_jobs) if available_jobs else 0
    
    return schedule

def display_schedule(schedule, title="勤務シフト結果"):
    """勤務シフト表を表示する汎用関数"""
    n_staff, n_day = schedule.shape
    
    # 勤務タイプのマッピング
    job_names = {
        0: "休み",
        1: "勤務A", 2: "勤務B",
        3: "早番A", 4: "早番B", 5: "早番C", 6: "早番D",
        7: "遅番A", 8: "遅番B", 9: "遅番C", 10: "遅番D",
        11: "夜勤A", 12: "夜勤B", 13: "夜勤C"
    }
    
    # 色のマッピング
    color_map = {
        0: "shift-休み",
        1: "shift-早番A", 2: "shift-早番A",
        3: "shift-早番A", 4: "shift-早番B", 5: "shift-早番C", 6: "shift-早番D",
        7: "shift-遅番A", 8: "shift-遅番B", 9: "shift-遅番C", 10: "遅番D",
        11: "shift-遅番A", 12: "shift-遅番B", 13: "shift-遅番C"
    }
    
    st.subheader(title)
    
    # HTMLテーブルを作成
    table_html = "<table style='width:80%; border-collapse: collapse; margin: 20px auto; font-size: 0.9rem;'>"
    
    # テーブルヘッダー
    table_html += "<tr style='background-color: #f8f9fa;'>"
    table_html += "<th style='padding: 8px; border: 1px solid #dee2e6; text-align: center; font-weight: bold; font-size: 1rem; color: #000;'>Staff</th>"
    for t in range(n_day):
        table_html += f"<th style='padding: 8px; border: 1px solid #dee2e6; text-align: center; font-weight: bold; font-size: 1rem; color: #000;'>Day{t+1}</th>"
    table_html += "</tr>"
    
    # データ行
    for i in range(n_staff):
        table_html += f"<tr>"
        table_html += f"<td style='padding: 8px; border: 1px solid #dee2e6; text-align: center; font-weight: bold; font-size: 0.95rem; color: #000; background-color: #f8f9fa;'>Staff{i}</td>"
        for t in range(n_day):
            job_id = schedule[i][t]
            job_name = job_names.get(job_id, f"Job{job_id}")
            color_class = color_map.get(job_id, "shift-休み")
            table_html += f"<td style='padding: 0; border: 1px solid #dee2e6; text-align: center;'><div class='shift-cell {color_class}' style='padding: 6px; font-size: 0.75rem;'>{job_name}</div></td>"
        table_html += "</tr>"
    
    table_html += "</table>"
    
    st.markdown(table_html, unsafe_allow_html=True)

def schedule_to_excel(schedule):
    """勤務シフト表をExcelファイルに変換する"""
    n_staff, n_day = schedule.shape
    
    # 勤務タイプのマッピング
    job_names = {
        0: "休み",
        1: "勤務A", 2: "勤務B",
        3: "早番A", 4: "早番B", 5: "早番C", 6: "早番D",
        7: "遅番A", 8: "遅番B", 9: "遅番C", 10: "遅番D",
        11: "夜勤A", 12: "夜勤B", 13: "夜勤C"
    }
    
    # DataFrameを作成
    data = []
    for i in range(n_staff):
        row = [f"Staff{i}"]
        for t in range(n_day):
            job_id = schedule[i][t]
            job_name = job_names.get(job_id, f"Job{job_id}")
            row.append(job_name)
        data.append(row)
    
    columns = ["Staff"] + [f"Day{t+1}" for t in range(n_day)]
    df = pd.DataFrame(data, columns=columns)
    
    # Excelファイルを作成
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='勤務シフト表', index=False)
    
    buffer.seek(0)
    return buffer

def main():
    """Streamlitによる新しいアプリ"""
    
    # カスタムスタイルを追加
    st.markdown("""
    <style>
        .shift-cell {
            padding: 0.5rem;
            text-align: center;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            color: white;
        }
        
        .shift-休み { background-color: #95a5a6; }
        .shift-早番A { background-color: #3498db; }
        .shift-早番B { background-color: #2980b9; }
        .shift-早番C { background-color: #1abc9c; }
        .shift-早番D { background-color: #16a085; }
        .shift-遅番A { background-color: #e74c3c; }
        .shift-遅番B { background-color: #c0392b; }
        .shift-遅番C { background-color: #f39c12; }
        .shift-遅番D { background-color: #d35400; }
        
        /* ファイルアップロードボタンのスタイルを変更 */
        .stFileUploader {
            width: 100% !important;
        }
        
        .stFileUploader > div {
            width: 100% !important;
        }
        
        .stFileUploader > div > div {
            width: 100% !important;
            height: 60px !important;
            border: 2px dashed #cccccc !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        .stFileUploader label {
            font-size: 16px !important;
            font-weight: bold !important;
        }
        
        /* --- 変更点：ボタンコンテナのスタイルを調整 --- */
        .long-button-container {
            width: 100%;
            margin: 10px 0;
        }
        
        .stButton > button {
            width: 100% !important;
            height: 60px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border-radius: 10px !important;
        }
        
        .download-button {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 60px;
            margin-top: 10px;
        }
        
        /* 小さいフォントのスタイル */
        .small-info {
            font-size: 14px !important;
            color: #666666 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    menu = ["Home","データ","モデル","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "データ":
        st.subheader("データ説明")
        st.write("💡 Excelフォーマットは管理者に問い合わせてください。")
        
        # サンプルデータファイルが存在するかチェック
        try:
            uploaded_xls = "optshift_sample2.xlsx"
            sheet = pd.read_excel(uploaded_xls, sheet_name=None, engine='openpyxl')
            st.success("✅ サンプルデータが正常に読み込まれました")
            
            # データ構造情報を表示
            st.write("**データシート構成:**")
            for sheet_name in sheet.keys():
                st.write(f"- {sheet_name}: {sheet[sheet_name].shape[0]}行 × {sheet[sheet_name].shape[1]}列")
        except FileNotFoundError:
            st.error("⚠️ サンプルデータファイル 'optshift_sample2.xlsx' が見つかりません")
        
        # 画像をチェックして表示
        try:
            from PIL import Image
            if os.path.exists('data.PNG'):
                image4 = Image.open('data.PNG')
                st.image(image4, use_column_width=True)
            else:
                st.info("📊 データ構造の説明図が利用できません")
                # テキスト形式でデータの説明を提供
                st.markdown("""
                ### データ構造説明
                
                **必要なExcelファイル構成:**
                - **day1シート**: 各日の勤務タイプ情報
                - **staff1シート**: スタッフの基本情報と制約
                - **jobシート**: 勤務種別の定義
                - **requirementシート**: 各勤務タイプの必要人数
                
                **主な列:**
                - day_type: 勤務日のタイプ
                - job_set: スタッフが対応可能な勤務
                - day_off: 休み希望日
                - requirement: 必要人数
                """)
        except ImportError:
            st.error("PIL ライブラリがインストールされていません")
        
    elif choice == "モデル":
        st.subheader("モデル説明")
        
        try:
            from PIL import Image
            # 各画像ファイルを個別にチェック
            image_files = ['mode3.PNG', 'mode1.PNG', 'mode2.PNG']
            image_loaded = False
            
            for img_file in image_files:
                if os.path.exists(img_file):
                    image = Image.open(img_file)
                    st.image(image, use_column_width=True)
                    image_loaded = True
                    
            if not image_loaded:
                st.info("📈 モデル説明図が利用できません")
                # テキスト形式でモデルの説明を提供
                st.markdown("""
                ### 最適化モデル概要
                
                **目的関数:**
                - 各制約の違反を最小化
                - 重み付きペナルティ方式
                
                **主要制約:**
                1. 📅 休み希望日出勤制約
                2. 🔄 連続勤務制限（4日・5日）
                3. 💤 連続休暇制限（4日）
                4. 👥 当日必要人数下限
                5. ⚡ 早番・遅番連続回避
                6. 🚫 スタッフ能力制約
                7. 🔀 選択的勤務制約
                8. 📊 月間休日数調整
                
                **求解手法:**
                - 遺伝的アルゴリズム + 局所探索
                - 集団サイズ: 50
                - 世代数: 最大100世代
                - 時間制限: 30秒
                """)
        except ImportError:
            st.error("PIL ライブラリがインストールされていません")
        
    elif choice == "About":
        st.subheader("アプリについて")
        st.write('張春来')
        st.write('東京海洋大学大学院　サプライチェーン最適化　数理最適化　')
        st.write('email: anlian0482@gmail.com')
    else:
        html_temp = """
        <div style="background-color:royalblue;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">🤖シフト・スケジューリングアプリ</h1>
        </div>
        """
        components.html(html_temp)
        
        # データ処理フラグを初期化
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        # ファイルアップロードボタン
        st.markdown('<div class="long-button-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader('📁 データファイルをアップロードしてください', type='xlsx', help="Excelファイルを選択してください")
        st.markdown('</div>', unsafe_allow_html=True)
        # 小さいフォントのヒント情報を追加
        st.markdown('<p class="small-info"> データフォーマットについては管理者にお問い合わせください（👇求解ボタンを押してサンプルデータで体験できます）</p>', unsafe_allow_html=True)
        
        # データ読み込みを処理
        load_data = False
        if uploaded_file is not None:
            sheet = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')
            load_data = True
            st.session_state.data_loaded = True
        else:
            # デフォルトでサンプルデータを使用
            try:
                uploaded_xls = "optshift_sample2.xlsx"
                sheet = pd.read_excel(uploaded_xls, sheet_name=None, engine='openpyxl')
                load_data = True
            except FileNotFoundError:
                st.error("⚠️ サンプルデータファイル 'optshift_sample2.xlsx' が見つかりません。データファイルをアップロードしてください。")
                st.markdown("""
                ### 必要なファイル形式
                
                **Excelファイル(.xlsx)に以下のシートが必要:**
                - `day1`: 各日の勤務タイプ情報
                - `staff1`: スタッフ情報と制約
                - `job`: 勤務種別定義  
                - `requirement`: 必要人数設定
                
                **サンプルファイルをお持ちでない場合は、管理者にお問い合わせください。**
                """)
                load_data = False
            
        if load_data:
            st.sidebar.title("⚙️ 重み")
            obj_weight=st.sidebar.slider("休み希望日出勤制約", 0, 100, 90)
            UB_max5_weight=st.sidebar.slider("５日連続勤務制約", 0, 100, 30)
            UB_max4_weight=st.sidebar.slider("4日連続勤務制約", 0, 100, 20)
            LB_min1_weight=st.sidebar.slider("４日連続休み制約", 0, 100, 10)
            LBC_weight=st.sidebar.slider("当日必要人数下限制約", 0, 100, 100)
            Disjective_weight=st.sidebar.slider("Staff1とStaff2のいずれかが勤務", 0, 100, 10)
            RestWorkRest_weight=st.sidebar.slider("休ー勤務ー休パターンの回避", 0, 100, 10)
            LateEarly_weight=st.sidebar.slider("遅番・早番の連続回避", 0, 80, 10)
            num_off_weight=st.sidebar.slider("月間休日数の調整", 0, 60, 10)
            
            # 重み辞書
            weights = {
                'obj_weight': obj_weight,
                'UB_max5_weight': UB_max5_weight,
                'UB_max4_weight': UB_max4_weight,
                'LB_min1_weight': LB_min1_weight,
                'LBC_weight': LBC_weight,
                'Disjective_weight': Disjective_weight,
                'RestWorkRest_weight': RestWorkRest_weight,
                'LateEarly_weight': LateEarly_weight,
                'num_off_weight': num_off_weight
            }
            
            month = 1 
            day_df = sheet["day"+str(month)]
            staff_df = sheet["staff"+str(month)]
            job_df = sheet["job"] 
            requirement_df = sheet["requirement"]
            
            # 15日間の勤務シフトに変更
            n_day = 15  # 15日に変更
            n_job = len(job_df)
            n_staff = 15
            
            # 早番、遅番のシフト
            early = [3,4,5,6] 
            late =  [7,8,9,10]
            # 月の休み
            num_off = 4
            # jobset
            job = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            
            # 要求タイプ、ジョブごとの必要人数を入れる辞書 requirement を準備
            requirement = defaultdict(int)
            for row in requirement_df.itertuples():
                requirement[row.day_type, row.job] = row.requirement
            
            # 休日希望日の集合を返す
            day_off = {}
            for i in range(n_staff):
                off = staff_df.loc[i, "day_off"]
                if pd.isnull(off):
                    day_off[i] = set([])
                else:
                    day_off[i] = set( ast.literal_eval(str(off)) )
                # 休日が15日の範囲を超えないようにする
                day_off[i] = {d for d in day_off[i] if d < n_day}
            
            # 担当しない勤務
            avoid_job = {1,2,12,13}
            job_set = {}
            for i in range(n_staff):
                job_set[i] = set(ast.literal_eval(staff_df.loc[i, "job_set"])) - avoid_job 
            
            # 必要人数下限 - 最初の15日間のみ取得
            LB = defaultdict(int)
            for t in range(n_day):  # 最初の15日間のみ処理
                if t < len(day_df):
                    row = day_df.iloc[t]
                    for j in job:
                        LB[t,j] = requirement[row.day_type, j]
                else:
                    # データが15日分ない場合は、最終日のデータを使用
                    last_row = day_df.iloc[-1]
                    for j in job:
                        LB[t,j] = requirement[last_row.day_type, j]
            
            # 最大休日日数
            B = {}
            for i in range(n_staff):
                B[i] = max(num_off, len(day_off[i]))
                # 休日日数が総日数を超えないようにする
                B[i] = min(B[i], n_day)
            
            # 各スタッフが担当できない勤務を定義
            avoid_jobs = {
                0: [1,2,4,5,7,8,9,11,12,13],
                1: [1,2,4,5,8,9,11,12,13],
                2: [1,2,5,8,9,11,12,13],
                3: [1,2,4,5,7,8,9,10,11,12,13],
                4: [1,2,3,5,7,8,9,11,12,13],
                5: [1,2,3,5,7,9,11,12,13],
                6: [1,2,3,5,9,11,12,13],
                7: [1,2,3,11,12,13],
                8: [1,2,3,11,12,13],
                9: [1,2,3,5,7,8,9,10,11,12,13],
                10: [1,2,3,5,7,8,9,10,11,12,13],
                11: [1,2,3,7,8,11,12,13],
                12: [1,2,3,7,11,12,13],
                13: [1,2,3,7,11,12,13],
                14: [1,2,3,7,8,11,12,13]
            }
            
            # 求解ボタン
            st.markdown('<div class="long-button-container">', unsafe_allow_html=True)
            if st.button("　　　　　　　　　　　　　🧬 遺伝的アルゴリズムで求解　　　　　　　　　　　　　　　", type="primary", help="遺伝的アルゴリズムを使用して勤務シフト計画を最適化します"):
                # プログレスバーを作成
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 遺伝的アルゴリズムソルバーを作成
                ga_solver = GeneticScheduler(
                    n_staff=n_staff,
                    n_day=n_day,
                    job=job,
                    requirement=requirement,
                    day_off=day_off,
                    avoid_jobs=avoid_jobs,
                    early=early,
                    late=late,
                    num_off=num_off,
                    B=B,
                    LB=LB,
                    weights=weights
                )
                
                # 進捗コールバック関数
                def progress_callback(progress, generation, fitness):
                    progress_bar.progress(progress)
                    status_text.text(f"進化中... 第{generation}世代, 現在の最良適応度: {fitness:.2f}")
                
                # 求解
                start_time = time.time()
                best_solution, best_fitness, generations = ga_solver.solve(
                    time_limit=30, 
                    progress_callback=progress_callback
                )
                end_time = time.time()
                
                # 完了後にプログレスバーを更新
                progress_bar.progress(1.0)
                status_text.text("✅ 求解完了！")
                
                if best_solution is not None:
                    st.session_state.optimized_schedule = best_solution
                    st.success(f"✅ 最適解が見つかりました！")
                    
                    # 求解統計情報を表示
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("求解時間", f"{end_time - start_time:.2f}秒")
                    with col_stat2:
                        st.metric("進世代数", generations)
                    with col_stat3:
                        st.metric("最良適応度", f"{best_fitness:.2f}")
                    
                    st.rerun()
                else:
                    st.error("❌ 求解に失敗しました。パラメータを調整して再試行してください")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ランダムな勤務シフト表を初期化して表示
            if 'initial_schedule' not in st.session_state:
                st.session_state.initial_schedule = create_random_schedule(n_staff, n_day, job, avoid_jobs, B)
            
            # 現在の勤務シフト表を表示（最適化結果があればそれを、なければランダム生成されたものを表示）
            if 'optimized_schedule' in st.session_state:
                display_schedule(st.session_state.optimized_schedule, "最適化後の勤務シフト結果")
            else:
                display_schedule(st.session_state.initial_schedule, "勤務シフト表")
                st.info("💡 上に表示されているのはランダムに生成された初期勤務シフト表です。「遺伝的アルゴリズムで求解」ボタンをクリックして、シフト計画を最適化してください！")
            
            # ダウンロードボタンを追加（最適化結果がある場合に表示）
            if 'optimized_schedule' in st.session_state:
                st.markdown('<div class="download-button">', unsafe_allow_html=True)
                excel_buffer = schedule_to_excel(st.session_state.optimized_schedule)
                st.download_button(
                    label="📥 勤務シフト表をダウンロード",
                    data=excel_buffer,
                    file_name="optimized_schedule.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="最適化された勤務シフト表をExcelファイルでダウンロード"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 最適化結果がある場合、統計情報を表示
            if 'optimized_schedule' in st.session_state:
                # 制約違反状況の統計
                st.subheader("📊 制約違反状況の統計")
                
                violations = {}
                best_solution = st.session_state.optimized_schedule
                
                # 1. 休み希望日出勤制約
                hope_violations = 0
                hope_total = 0
                for i in range(n_staff):
                    for t in range(n_day):
                        if t in day_off[i]:
                            hope_total += 1
                            if best_solution[i][t] != 0:
                                hope_violations += 1
                violations["休み希望日出勤制約"] = {"違反": hope_violations, "総数": hope_total}
                
                # 2. 5日連続勤務制約
                max5_violations = 0
                max5_total = 0
                for i in range(n_staff):
                    for t in range(n_day - 5):
                        max5_total += 1
                        consecutive_work = sum(1 for s in range(t, t + 6) if best_solution[i][s] != 0)
                        if consecutive_work > 5:
                            max5_violations += (consecutive_work - 5)
                violations["5日連続勤務制約"] = {"違反": max5_violations, "総数": max5_total}
                
                # 3. 4日連続勤務制約
                max4_violations = 0
                max4_total = 0
                for i in range(n_staff):
                    for t in range(n_day - 4):
                        max4_total += 1
                        consecutive_work = sum(1 for s in range(t, t + 5) if best_solution[i][s] != 0)
                        if consecutive_work > 4:
                            max4_violations += (consecutive_work - 4)
                violations["4日連続勤務制約"] = {"違反": max4_violations, "総数": max4_total}
                
                # 4. 4日連続休み制約
                rest_violations = 0
                rest_total = 0
                for i in range(n_staff):
                    for t in range(n_day - 3):
                        if t + 4 <= n_day:
                            rest_total += 1
                            consecutive_days = set(range(t, t + 4)) - day_off[i]
                            work_in_period = sum(1 for s in consecutive_days if s < n_day and best_solution[i][s] != 0)
                            if work_in_period == 0:
                                rest_violations += 1
                violations["4日連続休み制約"] = {"違反": rest_violations, "総数": rest_total}
                
                # 5. 当日必要人数下限制約
                requirement_violations = 0
                requirement_total = 0
                for t in range(n_day):
                    for j in job:
                        if j != 0:
                            requirement_total += 1
                            actual_count = sum(1 for i in range(n_staff) if best_solution[i][t] == j)
                            if actual_count < LB[t, j]:
                                requirement_violations += (LB[t, j] - actual_count)
                violations["当日必要人数下限制約"] = {"違反": requirement_violations, "総数": requirement_total}
                
                # 6. 勤務能力制約
                avoid_violations = 0
                avoid_total = n_staff * n_day
                for i in range(n_staff):
                    for t in range(n_day):
                        if best_solution[i][t] in avoid_jobs[i]:
                            avoid_violations += 1
                violations["勤務能力制約"] = {"違反": avoid_violations, "総数": avoid_total}
                
                # 7. Staff1・2選択勤務制約
                disjunctive_violations = 0
                disjunctive_total = n_day
                for t in range(n_day):
                    if best_solution[1][t] == 0 and best_solution[2][t] == 0:
                        disjunctive_violations += 1
                violations["Staff1・2選択勤務制約"] = {"違反": disjunctive_violations, "総数": disjunctive_total}
                
                # 8. 休-勤務-休パターン回避
                pattern_violations = 0
                pattern_total = 0
                for i in range(n_staff):
                    for t in range(n_day - 2):
                        pattern_total += 1
                        if (best_solution[i][t] == 0 and best_solution[i][t+1] != 0 and best_solution[i][t+2] == 0):
                            pattern_violations += 1
                violations["休-勤務-休パターン回避"] = {"違反": pattern_violations, "総数": pattern_total}
                
                # 9. 遅番・早番連続回避
                shift_violations = 0
                shift_total = 0
                for i in range(n_staff):
                    for t in range(n_day - 1):
                        shift_total += 1
                        if (best_solution[i][t] in early and best_solution[i][t+1] in late):
                            shift_violations += 1
                violations["遅番・早番連続回避"] = {"違反": shift_violations, "総数": shift_total}
                
                # 10. 月間休日数制約
                off_violations = 0
                off_total = n_staff
                for i in range(n_staff):
                    rest_days = sum(1 for t in range(n_day) if best_solution[i][t] == 0)
                    if rest_days != B[i]:
                        off_violations += abs(rest_days - B[i])
                violations["月間休日数制約"] = {"違反": off_violations, "総数": off_total}
                
                # 制約違反統計テーブルを作成
                constraint_data = []
                total_violations = 0
                total_constraints = 0
                
                for constraint_name, data in violations.items():
                    violation_count = data["違反"]
                    total_count = data["総数"]
                    satisfaction_rate = ((total_count - violation_count) / total_count * 100) if total_count > 0 else 100
                    status = "✅" if violation_count == 0 else "❌"
                    
                    constraint_data.append({
                        "制約タイプ": constraint_name,
                        "ステータス": status,
                        "違反数": violation_count,
                        "総制約数": total_count,
                        "充足率": f"{satisfaction_rate:.1f}%"
                    })
                    
                    total_violations += violation_count
                    total_constraints += total_count
                
                # 統計テーブルを表示
                constraint_df = pd.DataFrame(constraint_data)
                st.dataframe(constraint_df, use_container_width=True)
                
                # 全体的な品質評価を表示
                overall_satisfaction = ((total_constraints - total_violations) / total_constraints * 100) if total_constraints > 0 else 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("総制約数", total_constraints)
                with col2:
                    st.metric("総違反数", total_violations, delta=f"-{total_violations}" if total_violations > 0 else "0")
                with col3:
                    st.metric("全体充足率", f"{overall_satisfaction:.1f}%")
                with col4:
                    if total_violations == 0:
                        st.success("🎉 完璧なソリューションです！")
                    elif overall_satisfaction >= 80:
                        st.success("🟢 優れたソリューションです")
                    elif overall_satisfaction >= 70:
                        st.warning("🟡 良好なソリューションです")
                    else:
                        st.error("🔴 改善が必要です")

if __name__ == '__main__':
    main()
