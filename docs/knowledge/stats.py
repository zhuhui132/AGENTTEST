#!/usr/bin/env python3
"""
知识库统计脚本
"""

import os
import re
from pathlib import Path

def count_code_blocks(file_path):
    """统计代码块数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 统计代码块（```python 或 ```）
            code_blocks = len(re.findall(r'```python', content))
            return code_blocks
    except:
        return 0

def count_word_count(file_path):
    """统计字数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 移除代码块后再统计中文字数
            content_no_code = re.sub(r'```[\s\S]*?```', '', content)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content_no_code))
            return chinese_chars
    except:
        return 0

def analyze_knowledge_base():
    """分析知识库结构"""
    knowledge_dir = Path('.')

    stats = {
        'categories': {},
        'total_files': 0,
        'total_code_blocks': 0,
        'total_words': 0
    }

    # 遍历所有目录
    for category_dir in knowledge_dir.iterdir():
        if category_dir.is_dir() and category_dir.name not in ['.', '__pycache__']:
            category_name = category_dir.name
            category_stats = {
                'files': 0,
                'code_blocks': 0,
                'words': 0,
                'file_list': []
            }

            # 遍历目录中的文件
            for file_path in category_dir.glob('*.md'):
                if file_path.is_file():
                    category_stats['files'] += 1
                    code_blocks = count_code_blocks(file_path)
                    words = count_word_count(file_path)

                    category_stats['code_blocks'] += code_blocks
                    category_stats['words'] += words
                    category_stats['file_list'].append(file_path.name)

                    stats['total_code_blocks'] += code_blocks
                    stats['total_words'] += words

            stats['categories'][category_name] = category_stats
            stats['total_files'] += category_stats['files']

    return stats

def print_statistics(stats):
    """打印统计信息"""
    print("📚 AI知识库统计信息")
    print("=" * 50)
    print(f"📁 总分类数: {len(stats['categories'])}")
    print(f"📄 总文档数: {stats['total_files']}")
    print(f"💻 代码示例数: {stats['total_code_blocks']}")
    print(f"📝 总中文字数: {stats['total_words']:,}")
    print()

    print("📋 各分类统计:")
    print("-" * 50)

    for category, data in stats['categories'].items():
        print(f"🗂️ {category}")
        print(f"   📄 文档数: {data['files']}")
        print(f"   💻 代码示例: {data['code_blocks']}")
        print(f"   📝 中文字数: {data['words']:,}")

        if data['file_list']:
            print(f"   📋 文档列表:")
            for file in sorted(data['file_list']):
                print(f"      - {file}")
        print()

    print("🎯 主要技术覆盖:")
    print("-" * 50)

    # 技术领域统计
    tech_areas = {
        'machine-learning': '机器学习',
        'deep-learning': '深度学习',
        'llm': '大语言模型',
        'agents': '智能Agent',
        'reinforcement-learning': '强化学习',
        'multimodal': '多模态学习',
        'rag': '检索增强生成',
        'deployment': '模型部署',
        'security': '安全与伦理',
        'trends': '前沿技术'
    }

    for tech_key, tech_name in tech_areas.items():
        if tech_key in stats['categories']:
            data = stats['categories'][tech_key]
            print(f"  📖 {tech_name}: {data['files']}个文档, {data['code_blocks']}个代码示例")

def generate_index_content(stats):
    """生成目录内容"""
    content = """# 📚 AI知识库实时统计

## 📊 总体统计

| 指标 | 数量 |
|------|------|
| 分类数量 | {total_categories} |
| 文档总数 | {total_files} |
| 代码示例 | {total_code_blocks} |
| 中文字数 | {total_words:,} |

## 📋 分类详情

| 分类 | 文档数 | 代码示例 | 字数 |
|------|--------|----------|------|
""".format(
        total_categories=len(stats['categories']),
        total_files=stats['total_files'],
        total_code_blocks=stats['total_code_blocks'],
        total_words=stats['total_words']
    )

    for category, data in stats['categories'].items():
        category_names = {
            'machine-learning': '机器学习',
            'deep-learning': '深度学习',
            'llm': '大语言模型',
            'agents': '智能Agent',
            'reinforcement-learning': '强化学习',
            'multimodal': '多模态学习',
            'rag': '检索增强生成',
            'deployment': '模型部署',
            'security': '安全与伦理',
            'trends': '前沿技术'
        }

        category_name = category_names.get(category, category)
        content += f"| {category_name} | {data['files']} | {data['code_blocks']} | {data['words']:,} |\n"

    content += """

## 📈 更新记录

- `last_update`: 自动生成
- `version`: v3.0.0
- `status`: 活跃更新中

---

*此统计由脚本自动生成，反映当前知识库状态* 🚀
"""

    return content

if __name__ == "__main__":
    print("正在分析知识库结构...")
    stats = analyze_knowledge_base()

    print_statistics(stats)

    # 生成统计文档
    stats_content = generate_index_content(stats)
    with open('statistics.md', 'w', encoding='utf-8') as f:
        f.write(stats_content)

    print("\n📄 统计文档已生成: statistics.md")
    print("🚀 知识库分析完成！")
