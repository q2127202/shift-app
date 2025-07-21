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
sys.path.append('..')

# 遗传算法+局部搜索类
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
        
        # 遗传算法参数
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
    def create_individual(self):
        """创建一个个体（排班方案）"""
        individual = np.zeros((self.n_staff, self.n_day), dtype=int)
        
        for i in range(self.n_staff):
            # 先设置休息日
            rest_days = random.sample(range(self.n_day), self.B[i])
            for day in rest_days:
                individual[i][day] = 0
            
            # 为工作日分配工作
            work_days = [d for d in range(self.n_day) if d not in rest_days]
            available_jobs = [j for j in self.job if j != 0 and j not in self.avoid_jobs[i]]
            
            for day in work_days:
                if day in self.day_off[i]:
                    individual[i][day] = 0  # 希望休息的日子设为休息
                else:
                    individual[i][day] = random.choice(available_jobs) if available_jobs else 0
        
        return individual
    
    def calculate_fitness(self, individual):
        """计算个体的适应度（违反约束的惩罚分数，越低越好）"""
        penalty = 0
        
        # 1. 休み希望日出勤制约
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if t in self.day_off[i] and individual[i][t] != 0:
                    penalty += self.weights['obj_weight']
        
        # 2. 5日连续出勤制约
        for i in range(self.n_staff):
            for t in range(self.n_day - 5):
                consecutive_work = sum(1 for s in range(t, t + 6) if individual[i][s] != 0)
                if consecutive_work > 5:
                    penalty += self.weights['UB_max5_weight'] * (consecutive_work - 5)
        
        # 3. 4日连续出勤制约
        for i in range(self.n_staff):
            for t in range(self.n_day - 4):
                consecutive_work = sum(1 for s in range(t, t + 5) if individual[i][s] != 0)
                if consecutive_work > 4:
                    penalty += self.weights['UB_max4_weight'] * (consecutive_work - 4)
        
        # 4. 4日连续休み制约
        for i in range(self.n_staff):
            for t in range(self.n_day - 3):
                if t + 4 <= self.n_day:
                    consecutive_days = set(range(t, t + 4)) - self.day_off[i]
                    work_in_period = sum(1 for s in consecutive_days if s < self.n_day and individual[i][s] != 0)
                    if work_in_period == 0:
                        penalty += self.weights['LB_min1_weight']
        
        # 5. 当日必要人数下限满足
        for t in range(self.n_day):
            for j in self.job:
                if j != 0:
                    actual_count = sum(1 for i in range(self.n_staff) if individual[i][t] == j)
                    if actual_count < self.LB[t, j]:
                        penalty += self.weights['LBC_weight'] * (self.LB[t, j] - actual_count)
        
        # 6. 不能做的工作约束
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if individual[i][t] in self.avoid_jobs[i]:
                    penalty += 1000  # 严重违反
        
        # 7. Staff1和Staff2的选择性出勤
        for t in range(self.n_day):
            if individual[1][t] == 0 and individual[2][t] == 0:
                penalty += self.weights['Disjective_weight']
        
        # 8. 休-出勤-休みの回避
        for i in range(self.n_staff):
            for t in range(self.n_day - 2):
                if (individual[i][t] == 0 and individual[i][t+1] != 0 and individual[i][t+2] == 0):
                    penalty += self.weights['RestWorkRest_weight']
        
        # 9. 遅番・早番の连续回避
        for i in range(self.n_staff):
            for t in range(self.n_day - 1):
                if (individual[i][t] in self.early and individual[i][t+1] in self.late):
                    penalty += self.weights['LateEarly_weight']
        
        # 10. 月休日数约束
        for i in range(self.n_staff):
            rest_days = sum(1 for t in range(self.n_day) if individual[i][t] == 0)
            if rest_days != self.B[i]:
                penalty += self.weights['num_off_weight'] * abs(rest_days - self.B[i])
        
        return -penalty  # 负值，因为我们要最大化适应度
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择交叉点
        cross_point = random.randint(0, self.n_day - 1)
        
        for i in range(self.n_staff):
            # 交换交叉点后的基因
            child1[i][cross_point:] = parent2[i][cross_point:]
            child2[i][cross_point:] = parent1[i][cross_point:]
        
        return child1, child2
    
    def mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if random.random() < self.mutation_rate:
                    available_jobs = [j for j in self.job if j not in self.avoid_jobs[i]]
                    mutated[i][t] = random.choice(available_jobs)
        
        return mutated
    
    def local_search(self, individual):
        """局部搜索优化"""
        best = individual.copy()
        best_fitness = self.calculate_fitness(best)
        
        # 尝试改进每个员工的排班
        for i in range(self.n_staff):
            for t in range(self.n_day):
                if t not in self.day_off[i]:  # 不是希望休息的日子
                    available_jobs = [j for j in self.job if j not in self.avoid_jobs[i]]
                    current_job = individual[i][t]
                    
                    for job in available_jobs:
                        if job != current_job:
                            # 尝试新的工作安排
                            test_individual = individual.copy()
                            test_individual[i][t] = job
                            fitness = self.calculate_fitness(test_individual)
                            
                            if fitness > best_fitness:
                                best = test_individual.copy()
                                best_fitness = fitness
        
        return best
    
    def solve(self, time_limit=30):
        """使用遗传算法+局部搜索求解"""
        start_time = time.time()  # 记录开始时间
        
        # 初始化种群
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = float('-inf')
        
        generation = 0
        # 关键：在while循环条件中检查时间限制
        while generation < self.generations and (time.time() - start_time) < time_limit:
            # 计算适应度
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 更新最佳个体
            if fitness_scores[0][1] > best_fitness:
                best_individual = fitness_scores[0][0].copy()
                best_fitness = fitness_scores[0][1]
            
            # 选择精英
            elite = [ind for ind, _ in fitness_scores[:self.elite_size]]
            
            # 生成新一代
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # 锦标赛选择
                tournament_size = 3
                parents = random.sample(fitness_scores[:self.population_size//2], 2)
                parent1, parent2 = parents[0][0], parents[1][0]
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            generation += 1
            
            # 每10代进行一次局部搜索
            if generation % 10 == 0 and best_individual is not None:
                improved = self.local_search(best_individual)
                improved_fitness = self.calculate_fitness(improved)
                if improved_fitness > best_fitness:
                    best_individual = improved
                    best_fitness = improved_fitness
        
        return best_individual, best_fitness, generation

def main():
    """a new app with Streamlit"""
    
    menu = ["Home","データ","モデル","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "データ":
        st.subheader("データ説明")
        uploaded_xls = "optshift_sample2.xlsx"
        sheet = pd.read_excel(uploaded_xls, sheet_name=None, engine='openpyxl')
        print(sheet)
        
        from PIL import Image
        image4 = Image.open('data.PNG')
        st.image(image4,use_column_width=True)    
        
    elif choice == "モデル":
        from PIL import Image
        image2 = Image.open('mode3.PNG')
        st.image(image2,use_column_width=True)    
        image = Image.open('mode1.PNG')
        st.image(image,use_column_width=True)
        image1 = Image.open('mode2.PNG')
        st.image(image1,use_column_width=True)
        
    elif choice == "About":
        st.subheader("About App")
        st.write('張春来')
        st.write('東京海洋大学大学院　サプライチェーン最適化　数理最適化　')
        st.write('email: anlian0482@gmail.com')
    else:
        st.subheader("Home")
        html_temp = """
        <div style="background-color:royalblue;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">シフト・スケジューリング　アプリ</h1>
        </div>
        """
        
        components.html(html_temp)
        
        uploaded_file = st.file_uploader('1. データファイルをアップロードしてください。', type='xlsx')
        
        check = st.checkbox('サンプルデータを使います', value=False)
        
        if uploaded_file is not None:
            if 'push1' not in st.session_state:
                st.session_state.push1 = False
                
            button1 = st.button(' ファイル読み込み')
            
            if button1:
                st.session_state.push1 = True
                
        if (uploaded_file is not None and st.session_state.push1) or check:
            
            st.sidebar.title("⚙️ 重み")
            obj_weight=st.sidebar.slider("休み希望日出勤制約", 0, 100, 90)
            UB_max5_weight=st.sidebar.slider("５日連続出勤制約", 0, 100, 30)
            UB_max4_weight=st.sidebar.slider("4日連続出勤制約", 0, 100, 20)
            LB_min1_weight=st.sidebar.slider("４日連続休み制約", 0, 100, 10)
            LBC_weight=st.sidebar.slider("当日必要人数下限満足", 0, 100, 100)
            Disjective_weight=st.sidebar.slider("Staff1とStaff2のいずれかが出勤", 0, 100, 10)
            RestWorkRest_weight=st.sidebar.slider("休ー出勤ー休みの回避", 0, 100, 10)
            LateEarly_weight=st.sidebar.slider("遅番・早番の連続回避", 0, 80, 10)
            num_off_weight=st.sidebar.slider("月休日最大化", 0, 60, 10)
            
            # 权重字典
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
            
            if uploaded_file is not None and st.session_state.push1:
                sheet = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')
                if button1:
                    st.write(sheet)
            else:
                uploaded_xls = "optshift_sample2.xlsx"
                sheet = pd.read_excel(uploaded_xls, sheet_name=None, engine='openpyxl')
            
            month = 1 
            day_df = sheet["day"+str(month)]
            staff_df = sheet["staff"+str(month)]
            job_df = sheet["job"] 
            requirement_df = sheet["requirement"]
            
            # 修改为15天排班
            n_day = 15  # 改为15天
            n_job = len(job_df)
            n_staff = 15
            
            # 早番，遅番のシフト
            early = [3,4,5,6] 
            late =  [7,8,9,10]
            # 月の休み
            num_off = 9 
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
                # 确保休假日不超过15天范围
                day_off[i] = {d for d in day_off[i] if d < n_day}
            
            # 避免工作
            avoid_job = {1,2,12,13}
            job_set = {}
            for i in range(n_staff):
                job_set[i] = set(ast.literal_eval(staff_df.loc[i, "job_set"])) - avoid_job 
            
            # 必要人数下限 - 只取前15天
            LB = defaultdict(int)
            for t in range(n_day):  # 只处理前15天
                if t < len(day_df):
                    row = day_df.iloc[t]
                    for j in job:
                        LB[t,j] = requirement[row.day_type, j]
                else:
                    # 如果数据不足15天，使用最后一天的数据
                    last_row = day_df.iloc[-1]
                    for j in job:
                        LB[t,j] = requirement[last_row.day_type, j]
            
            # 最大休み日数
            B = {}
            for i in range(n_staff):
                B[i] = max(num_off, len(day_off[i]))
                # 确保休假日数不超过总天数
                B[i] = min(B[i], n_day)
            
            # 定义每个员工不能做的工作
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
            
            # 创建求解按钮
            if st.button("遗传算法求解"):
                with st.spinner("正在使用遗传算法+局部搜索求解中..."):
                    # 创建遗传算法求解器
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
                    
                    # 求解
                    start_time = time.time()
                    best_solution, best_fitness, generations = ga_solver.solve(time_limit=30)
                    end_time = time.time()
                    
                    st.success(f"求解完成！")
                    st.write(f"求解时间: {end_time - start_time:.2f}秒")
                    st.write(f"进化代数: {generations}")
                    st.write(f"最佳适应度: {best_fitness}")
                    
                    if best_solution is not None:
                        # 显示结果
                        st.subheader("排班结果")
                        
                        # 创建结果DataFrame
                        result_df = pd.DataFrame(best_solution)
                        result_df.index = [f"Staff{i}" for i in range(n_staff)]
                        result_df.columns = [f"Day{i+1}" for i in range(n_day)]
                        
                        st.dataframe(result_df)
                        
                        # 约束违约情况统计
                        st.subheader("约束违约情况统计")
                        
                        violations = {}
                        
                        # 1. 休み希望日出勤违约
                        hope_violations = 0
                        hope_total = 0
                        for i in range(n_staff):
                            for t in range(n_day):
                                if t in day_off[i]:
                                    hope_total += 1
                                    if best_solution[i][t] != 0:
                                        hope_violations += 1
                        violations["休み希望日出勤制約"] = {"违约": hope_violations, "总数": hope_total}
                        
                        # 2. 5日连续出勤违约
                        max5_violations = 0
                        max5_total = 0
                        for i in range(n_staff):
                            for t in range(n_day - 5):
                                max5_total += 1
                                consecutive_work = sum(1 for s in range(t, t + 6) if best_solution[i][s] != 0)
                                if consecutive_work > 5:
                                    max5_violations += (consecutive_work - 5)
                        violations["5日连続出勤制約"] = {"违约": max5_violations, "总数": max5_total}
                        
                        # 3. 4日连续出勤违约
                        max4_violations = 0
                        max4_total = 0
                        for i in range(n_staff):
                            for t in range(n_day - 4):
                                max4_total += 1
                                consecutive_work = sum(1 for s in range(t, t + 5) if best_solution[i][s] != 0)
                                if consecutive_work > 4:
                                    max4_violations += (consecutive_work - 4)
                        violations["4日连続出勤制約"] = {"违约": max4_violations, "总数": max4_total}
                        
                        # 4. 4日连续休み违约
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
                        violations["4日連続休み制約"] = {"违约": rest_violations, "总数": rest_total}
                        
                        # 5. 当日必要人数下限违约
                        requirement_violations = 0
                        requirement_total = 0
                        for t in range(n_day):
                            for j in job:
                                if j != 0:
                                    requirement_total += 1
                                    actual_count = sum(1 for i in range(n_staff) if best_solution[i][t] == j)
                                    if actual_count < LB[t, j]:
                                        requirement_violations += (LB[t, j] - actual_count)
                        violations["当日必要人数下限制約"] = {"违约": requirement_violations, "总数": requirement_total}
                        
                        # 6. 不能做的工作违约
                        avoid_violations = 0
                        avoid_total = n_staff * n_day
                        for i in range(n_staff):
                            for t in range(n_day):
                                if best_solution[i][t] in avoid_jobs[i]:
                                    avoid_violations += 1
                        violations["工作能力制約"] = {"违约": avoid_violations, "总数": avoid_total}
                        
                        # 7. Staff1和Staff2选择性出勤违约
                        disjunctive_violations = 0
                        disjunctive_total = n_day
                        for t in range(n_day):
                            if best_solution[1][t] == 0 and best_solution[2][t] == 0:
                                disjunctive_violations += 1
                        violations["Staff1・2選択出勤制約"] = {"违约": disjunctive_violations, "总数": disjunctive_total}
                        
                        # 8. 休-出勤-休み违约
                        pattern_violations = 0
                        pattern_total = 0
                        for i in range(n_staff):
                            for t in range(n_day - 2):
                                pattern_total += 1
                                if (best_solution[i][t] == 0 and best_solution[i][t+1] != 0 and best_solution[i][t+2] == 0):
                                    pattern_violations += 1
                        violations["休-出勤-休パターン回避"] = {"违约": pattern_violations, "总数": pattern_total}
                        
                        # 9. 遅番・早番连续违约
                        shift_violations = 0
                        shift_total = 0
                        for i in range(n_staff):
                            for t in range(n_day - 1):
                                shift_total += 1
                                if (best_solution[i][t] in early and best_solution[i][t+1] in late):
                                    shift_violations += 1
                        violations["遅番・早番連続回避"] = {"违约": shift_violations, "总数": shift_total}
                        
                        # 10. 月休日数违约
                        off_violations = 0
                        off_total = n_staff
                        for i in range(n_staff):
                            rest_days = sum(1 for t in range(n_day) if best_solution[i][t] == 0)
                            if rest_days != B[i]:
                                off_violations += abs(rest_days - B[i])
                        violations["月休日数制約"] = {"违约": off_violations, "总数": off_total}
                        
                        # 创建约束违约统计表格
                        constraint_data = []
                        total_violations = 0
                        total_constraints = 0
                        
                        for constraint_name, data in violations.items():
                            violation_count = data["违约"]
                            total_count = data["总数"]
                            satisfaction_rate = ((total_count - violation_count) / total_count * 100) if total_count > 0 else 100
                            status = "✅" if violation_count == 0 else "❌"
                            
                            constraint_data.append({
                                "约束类型": constraint_name,
                                "状态": status,
                                "违约数": violation_count,
                                "总约束数": total_count,
                                "满足率": f"{satisfaction_rate:.1f}%"
                            })
                            
                            total_violations += violation_count
                            total_constraints += total_count
                        
                        # 显示统计表格
                        constraint_df = pd.DataFrame(constraint_data)
                        st.dataframe(constraint_df, use_container_width=True)
                        
                        # 显示总体质量评估
                        overall_satisfaction = ((total_constraints - total_violations) / total_constraints * 100) if total_constraints > 0 else 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总约束数", total_constraints)
                        with col2:
                            st.metric("总违约数", total_violations, delta=f"-{total_violations}" if total_violations > 0 else "0")
                        with col3:
                            st.metric("整体满足率", f"{overall_satisfaction:.1f}%")
                        with col4:
                            if total_violations == 0:
                                st.success("🎉 完美解决方案！")
                            elif overall_satisfaction >= 80:
                                st.success("🟢 优秀解决方案")
                            elif overall_satisfaction >= 70:
                                st.warning("🟡 良好解决方案")
                            else:
                                st.error("🔴 需要改进")
                    else:
                        st.error("求解失败，请调整参数后重试")

if __name__ == '__main__':
    main()
