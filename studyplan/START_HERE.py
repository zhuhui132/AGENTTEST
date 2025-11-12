#!/usr/bin/env python3
"""
AI学习系统启动器

一键启动完整的AI学习体验，包括路径推荐、交互式问答和进度跟踪。

作者: AI学习团队
版本: 1.0.0
日期: 2025-11-12
"""

import os
import sys
from learning_path_finder import LearningPathFinder
from interactive_qa import InteractiveQA
from progress_tracker import LearningProgressTracker


def main():
    """主启动器"""
    print("=" * 70)
    print("🎓 AI技术学习系统 - 启动器")
    print("=" * 70)
    print("📚 知识库覆盖：从1943年神经网络基础到2025年AI前沿")
    print("🛠️  实践验证：200+测试用例验证理论知识")
    print("🎯 个性化路径：根据背景和目标定制学习计划")
    print("📊 进度跟踪：实时监控学习进度和成果")
    print("🤝 智能问答：24/7学习助手随时答疑")
    print("=" * 70)

    # 显示菜单
    print("\n🎯 请选择功能:")
    print("1. 📋 查看学习路径规划")
    print("2. 💬 启动交互式问答系统")
    print("3. 📊 学习进度跟踪")
    print("4. 🚀 开始全新学习体验")
    print("5. ❓ 查看使用帮助")
    print("0. 🚪 退出系统")

    while True:
        try:
            choice = input("\n👤 请输入选择 (0-5): ").strip()

            if choice == "0":
                print("\n👋 学习愉快，再见！")
                break
            elif choice == "1":
                learning_path_planner()
            elif choice == "2":
                start_qa_system()
            elif choice == "3":
                progress_tracking_system()
            elif choice == "4":
                start_full_experience()
            elif choice == "5":
                show_help()
            else:
                print("❌ 无效选择，请输入0-5之间的数字")

        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            print("💡 请重新选择或联系管理员")


def learning_path_planner():
    """学习路径规划"""
    print("\n" + "=" * 60)
    print("📋 学习路径规划器")
    print("=" * 60)

    # 收集用户信息
    print("\n🎯 帮助我们了解你的学习需求:")

    name = input("📝 你的姓名: ").strip()
    if not name:
        name = "学习者"

    print("\n💻 编程技能水平:")
    print("1. 初级 - 刚开始学习编程")
    print("2. 中级 - 有一定编程经验")
    print("3. 高级 - 编程经验丰富")
    print("4. 专家 - 专业开发者")
    skill_choice = input("👤 请选择 (1-4): ").strip()
    skill_levels = {"1": "初级", "2": "中级", "3": "高级", "4": "专家"}
    skill_level = skill_levels.get(skill_choice, "中级")

    print("\n🎯 学习目标:")
    print("1. research - AI研究")
    print("2. engineering - AI工程实践")
    print("3. product - AI产品开发")
    print("4. beginner - AI基础入门")
    goal_choice = input("🎯 请选择 (1-4): ").strip()
    goal_map = {"1": "research", "2": "engineering", "3": "product", "4": "beginner"}
    learning_goal = goal_map.get(goal_choice, "engineering")

    time_commitment = input("⏰ 每周可投入学习时间 (小时): ").strip()
    try:
        time_commitment = float(time_commitment)
        if time_commitment < 1:
            time_commitment = 10  # 默认值
    except:
        time_commitment = 10

    print("\n📚 教育背景:")
    print("1. 计算机科学")
    print("2. 数学/统计")
    print("3. 软件工程")
    print("4. 其他工程")
    print("5. 文科/商科")
    bg_choice = input("📖 请选择 (1-5): ").strip()
    background_map = {
        "1": "计算机科学", "2": "数学统计",
        "3": "软件工程", "4": "其他工程", "5": "文科商科"
    }
    background = background_map.get(bg_choice, "计算机科学")

    # 创建用户档案
    user_profile = {
        "name": name,
        "skill_level": skill_level,
        "learning_goal": learning_goal,
        "time_commitment": time_commitment,
        "background": background
    }

    # 获取推荐路径
    finder = LearningPathFinder()
    result = finder.find_best_path(user_profile)

    print("\n" + "=" * 60)
    print("🎯 学习路径推荐结果")
    print("=" * 60)
    print(f"👋 推荐路径: {result['recommended_path']}")
    print(f"📊 匹配置信度: {result['recommendation_confidence']:.1f}")
    print(f"📝 路径描述: {result['path_info']['description']}")
    print(f"⏰ 预计时间: {result['path_info']['duration']}")

    print(f"\n🎯 备选路径:")
    for alt in result['alternative_paths']:
        print(f"  • {alt}")

    # 生成个性化计划
    plan = finder.generate_personalized_plan(user_profile, result['path_info'])

    print(f"\n📋 个性化学习计划:")
    print(f"📅 学习周期: {plan['duration_weeks']}周")
    print(f"⏰ 每周投入: {plan['time_commitment_weekly']}小时")
    print(f"🗓️ 开始日期: {plan['start_date']}")
    print(f"🗓️ 结束日期: {plan['end_date']}")

    print(f"\n📊 学习里程碑 (前3个):")
    for milestone in plan['milestones'][:3]:
        print(f"  • 第{milestone['week']}周: {milestone['milestone']}")

    print(f"\n📚 推荐资源:")
    for resource_type, resources in plan['recommended_resources'].items():
        print(f"  • {resource_type}: {len(resources)}项资源")

    # 询问是否要保存计划
    save = input(f"\n💾 是否保存学习计划到文件? (y/n): ").strip().lower()
    if save in ['y', 'yes']:
        filename = f"{name}_learning_plan.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"AI学习计划 - {name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"推荐路径: {result['recommended_path']}\n")
            f.write(f"置信度: {result['recommendation_confidence']:.1f}\n")
            f.write(f"学习周期: {plan['duration_weeks']}周\n")
            f.write(f"每周投入: {plan['time_commitment_weekly']}小时\n\n")
            f.write("详细计划请查看系统生成\n")
        print(f"✅ 计划已保存到: {filename}")

    # 询问是否要创建进度跟踪账户
    create_progress = input(f"\n📊 是否创建学习进度跟踪账户? (y/n): ").strip().lower()
    if create_progress in ['y', 'yes']:
        tracker = LearningProgressTracker()
        user_id = tracker.create_user(
            name=name,
            skill_level=skill_level,
            learning_goal=learning_goal,
            target_weeks=plan['duration_weeks']
        )
        print(f"✅ 进度跟踪账户已创建，用户ID: {user_id}")
        print("💡 请保存用户ID，用于后续进度查询")


def start_qa_system():
    """启动问答系统"""
    print("\n" + "=" * 60)
    print("💬 交互式问答系统")
    print("=" * 60)
    print("🎯 支持的问题类型:")
    print("  • 概念理解 - 什么是Transformer？")
    print("  • 技术实现 - 如何实现注意力机制？")
    print("  • 实践操作 - 如何运行测试？")
    print("  • 学习建议 - 如何选择学习路径？")
    print("  • 职业发展 - AI职业发展方向？")
    print("\n💡 输入 'help' 查看更多帮助")
    print("💡 输入 'quit' 退出问答系统")
    print("=" * 60)

    qa_system = InteractiveQA()
    qa_system.start_session()


def progress_tracking_system():
    """进度跟踪系统"""
    print("\n" + "=" * 60)
    print("📊 学习进度跟踪系统")
    print("=" * 60)

    tracker = LearningProgressTracker()

    while True:
        print("\n📋 功能选项:")
        print("1. 📝 创建学习账户")
        print("2. 📊 查看学习进度")
        print("3. 📝 记录学习活动")
        print("4. 🧪 记录测试结果")
        print("5. 🏆 标记学习里程碑")
        print("6. 📈 生成进度报告")
        print("7. 📊 生成进度图表")
        print("0. 🔙 返回主菜单")

        choice = input("👤 请选择功能 (0-7): ").strip()

        if choice == "0":
            break
        elif choice == "1":
            create_user_account(tracker)
        elif choice == "2":
            view_progress(tracker)
        elif choice == "3":
            record_learning_activity(tracker)
        elif choice == "4":
            record_test_result(tracker)
        elif choice == "5":
            record_milestone(tracker)
        elif choice == "6":
            generate_report(tracker)
        elif choice == "7":
            generate_chart(tracker)
        else:
            print("❌ 无效选择，请输入0-7之间的数字")


def create_user_account(tracker):
    """创建用户账户"""
    print("\n📝 创建学习账户")

    name = input("👤 姓名: ").strip()
    if not name:
        print("❌ 姓名不能为空")
        return

    skill_level = input("💻 技能水平 (初级/中级/高级): ").strip()
    learning_goal = input("🎯 学习目标 (research/engineering/product/beginner): ").strip()

    try:
        target_weeks = int(input("📅 学习周期 (周): ").strip())
    except:
        target_weeks = 8

    user_id = tracker.create_user(
        name=name,
        skill_level=skill_level,
        learning_goal=learning_goal,
        target_weeks=target_weeks
    )

    print(f"✅ 账户创建成功！")
    print(f"📋 用户ID: {user_id}")
    print("💡 请保存用户ID，用于后续操作")


def view_progress(tracker):
    """查看学习进度"""
    print("\n📊 查看学习进度")

    try:
        user_id = int(input("👤 请输入用户ID: ").strip())
    except:
        print("❌ 用户ID必须是数字")
        return

    progress = tracker.get_user_progress(user_id)

    if "error" in progress:
        print(f"❌ {progress['error']}")
        return

    user_info = progress["user_info"]
    learning_info = progress["learning_info"]
    test_info = progress["test_info"]
    stage_progress = progress["stage_progress"]

    print(f"\n👋 学习者: {user_info['name']}")
    print(f"📚 学习目标: {user_info['learning_goal']}")
    print(f"⏰ 开始日期: {user_info['start_date'][:10]}")
    print(f"📊 总学习时间: {learning_info['total_hours']:.1f}小时")
    print(f"🧪 平均测试得分: {test_info['avg_score']:.1f}%")

    print(f"\n📈 各阶段进度:")
    for stage, progress_data in stage_progress.items():
        stage_emoji = {"基础理论": "📖", "核心技术": "🔬", "工程实践": "🛠️", "应用创新": "🚀"}.get(stage, "📌")
        print(f"  {stage_emoji} {stage}: {progress_data['completion_rate']:.1f}%")


def record_learning_activity(tracker):
    """记录学习活动"""
    print("\n📝 记录学习活动")

    try:
        user_id = int(input("👤 请输入用户ID: ").strip())
    except:
        print("❌ 用户ID必须是数字")
        return

    date = input("📅 学习日期 (YYYY-MM-DD, 按回车使用今天): ").strip()
    if not date:
        from datetime import datetime
        date = datetime.now().strftime("%Y-%m-%d")

    topic = input("📚 学习主题: ").strip()

    try:
        hours = float(input("⏰ 学习时间 (小时): ").strip())
        completion = float(input("📊 完成度 (%): ").strip())
    except:
        print("❌ 时间和完成度必须是数字")
        return

    notes = input("📝 学习笔记 (可选): ").strip()

    success = tracker.add_learning_record(user_id, date, topic, hours, completion, notes)

    if success:
        print("✅ 学习记录添加成功！")
    else:
        print("❌ 学习记录添加失败")


def record_test_result(tracker):
    """记录测试结果"""
    print("\n🧪 记录测试结果")

    try:
        user_id = int(input("👤 请输入用户ID: ").strip())
    except:
        print("❌ 用户ID必须是数字")
        return

    test_name = input("🧪 测试名称: ").strip()

    try:
        total = int(input("📊 总题目数: ").strip())
        passed = int(input("✅ 通过题目数: ").strip())
    except:
        print("❌ 题目数必须是数字")
        return

    execution_time = input("⏰ 执行时间 (可选): ").strip()

    success = tracker.add_test_record(user_id, test_name, total, passed, execution_time)

    if success:
        score = (passed / total * 100) if total > 0 else 0
        print(f"✅ 测试记录添加成功！得分: {score:.1f}%")
    else:
        print("❌ 测试记录添加失败")


def record_milestone(tracker):
    """记录学习里程碑"""
    print("\n🏆 标记学习里程碑")

    try:
        user_id = int(input("👤 请输入用户ID: ").strip())
    except:
        print("❌ 用户ID必须是数字")
        return

    print("📊 里程碑类别:")
    print("1. 基础理论")
    print("2. 核心技术")
    print("3. 工程实践")
    print("4. 应用创新")

    category_choice = input("🎯 请选择类别 (1-4): ").strip()
    categories = {"1": "基础理论", "2": "核心技术", "3": "工程实践", "4": "应用创新"}
    category = categories.get(category_choice, "基础理论")

    name = input("🏆 里程碑名称: ").strip()
    notes = input("📝 备注 (可选): ").strip()

    success = tracker.add_milestone(user_id, category, name, notes)

    if success:
        print("✅ 里程碑记录成功！")
    else:
        print("❌ 里程碑记录失败")


def generate_report(tracker):
    """生成进度报告"""
    print("\n📈 生成学习进度报告")

    try:
        user_id = int(input("👤 请输入用户ID: ").strip())
    except:
        print("❌ 用户ID必须是数字")
        return

    report = tracker.generate_progress_report(user_id)

    filename = f"progress_report_user_{user_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 进度报告已生成: {filename}")

    # 显示报告摘要
    lines = report.split('\n')[:20]  # 显示前20行
    print("\n📊 报告摘要:")
    for line in lines:
        if line.strip():
            print(line)
    print("...\n💡 完整报告请查看文件")


def generate_chart(tracker):
    """生成进度图表"""
    print("\n📊 生成学习进度图表")

    try:
        user_id = int(input("👤 请输入用户ID: ").strip())
    except:
        print("❌ 用户ID必须是数字")
        return

    result = tracker.generate_progress_chart(user_id)

    print(f"📈 {result}")

    # 尝试显示图表（如果matplotlib可用）
    try:
        import matplotlib.pyplot as plt
        print("💡 图表文件已生成，可以使用图片查看器查看")
    except ImportError:
        print("💡 请安装matplotlib以查看图表: pip install matplotlib")


def start_full_experience():
    """开始完整学习体验"""
    print("\n" + "=" * 60)
    print("🚀 完整学习体验")
    print("=" * 60)
    print("🎯 这是为学习者设计的综合体验")
    print("📚 包括路径规划、问答、进度跟踪的完整流程")
    print("=" * 60)

    # 第一步：学习路径规划
    print("📋 第一步：学习路径规划")
    learning_path_planner()

    input("\n⏸️ 按回车继续到第二步...")

    # 第二步：演示问答系统
    print("\n💬 第二步：体验问答系统")
    print("💡 这是一个演示，你可以输入示例问题:")
    print("   • 什么是Transformer？")
    print("   • 如何运行测试？")
    print("   • 输入 'quit' 退出")

    # 创建一个简单的问答演示
    print("\n🤔 请输入你的问题 (demo模式):")
    question = input("❓ 问题: ").strip()

    if question:
        # 模拟回答
        print("🤖 AI助手: 这是一个演示回答。")
        print("📚 完整的问答系统请在主菜单选择选项2")

    input("\n⏸️ 按回车继续到第三步...")

    # 第三步：演示进度跟踪
    print("\n📊 第三步：进度跟踪演示")
    print("💡 完整的进度跟踪请在主菜单选择选项3")
    print("📋 这里演示如何记录学习数据")

    # 创建一个演示用户
    tracker = LearningProgressTracker()
    demo_user_id = tracker.create_user(
        name="演示用户",
        skill_level="中级",
        learning_goal="engineering",
        target_weeks=8
    )

    print(f"✅ 创建演示用户: {demo_user_id}")

    # 添加一些示例数据
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    tracker.add_learning_record(demo_user_id, today, "Transformer原理", 2.5, 80.0, "学习注意力机制")
    tracker.add_test_record(demo_user_id, "test_transformer.py", 15, 12, "10:30")
    tracker.add_milestone(demo_user_id, "核心技术", "掌握注意力机制", "完成学习笔记")

    # 生成报告
    report = tracker.generate_progress_report(demo_user_id)
    print("📈 演示用户进度报告:")
    print("=" * 40)
    print(report[:500] + "...")  # 显示前500字符
    print("=" * 40)

    filename = f"demo_progress_report.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 完整演示报告已保存到: {filename}")

    input("\n⏸️ 按回车返回主菜单...")


def show_help():
    """显示帮助信息"""
    print("\n" + "=" * 60)
    print("📖 AI学习系统使用帮助")
    print("=" * 60)

    print("\n🎯 系统功能说明:")
    print("1. 📋 学习路径规划 - 根据你的背景和目标推荐最适合的学习路径")
    print("2. 💬 交互式问答 - 智能问答系统，随时解答学习问题")
    print("3. 📊 进度跟踪 - 记录学习进度，生成详细报告和图表")
    print("4. 🚀 完整体验 - 一站式学习体验，整合所有功能")

    print("\n📚 知识库覆盖:")
    print("• AI发展历程 (1943-2025)")
    print("• 核心技术原理")
    print("• 工程实践指南")
    print("• 测试验证用例")

    print("\n💡 使用建议:")
    print("• 新手建议从选项1开始，先制定学习计划")
    print("• 学习过程中随时使用选项2提问答疑")
    print("• 定期使用选项3跟踪和调整学习进度")
    print("• 选项4适合想体验完整流程的学习者")

    print("\n🛠️ 技术要求:")
    print("• Python 3.7+")
    print("• 推荐安装 matplotlib (用于图表生成)")
    print("• 支持UTF-8编码的环境")

    print("\n📞 获取帮助:")
    print("• 在使用中遇到问题，可以随时在问答系统中提问")
    print("• 查看项目文档了解更多功能")
    print("• 提交issue反馈问题和建议")

    print("\n🚀 快速开始:")
    print("1. 选择选项1制定个人学习计划")
    print("2. 按照推荐路径开始学习")
    print("3. 使用选项2解决学习中的问题")
    print("4. 使用选项3跟踪学习进度")
    print("5. 持续学习，达成目标！")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
