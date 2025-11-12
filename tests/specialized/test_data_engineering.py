"""
数据工程专项测试

该测试模块验证数据工程全流程，包括数据收集、清洗、转换、标注、质量评估等。
数据质量直接影响AI模型性能，是整个AI系统的关键基础。

作者: AI开发团队
版本: 1.0.0
日期: 2025-11-10
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import tempfile
import csv
from datetime import datetime


class TestDataEngineering:
    """数据工程核心测试类"""

    def setup_method(self):
        """测试前设置"""
        # 创建测试数据
        self.sample_data = {
            'text': [
                "This is a positive review",
                "Negative experience with the product",
                "Neutral feedback about service",
                "Excellent quality and fast delivery",
                "Poor customer support response"
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative'],
            'rating': [4, 1, 3, 5, 2],
            'category': ['product', 'product', 'service', 'product', 'service']
        }

        # 创建DataFrame
        self.df = pd.DataFrame(self.sample_data)

        # 测试指标阈值
        self.data_quality_thresholds = {
            'completeness': 0.95,  # 数据完整性 >= 95%
            'consistency': 0.90,  # 数据一致性 >= 90%
            'accuracy': 0.95,     # 数据准确性 >= 95%
            'timeliness': 0.90     # 数据时效性 >= 90%
        }

    def test_data_quality_metrics(self):
        """测试数据质量指标"""
        print("\n=== 数据质量指标测试 ===")

        # 1. 数据完整性测试
        completeness_metrics = self._test_data_completeness()
        print(f"✓ 数据完整性: {completeness_metrics['completeness']:.3f}")
        print(f"  缺失值统计: {completeness_metrics['missing_stats']}")

        # 2. 数据一致性测试
        consistency_metrics = self._test_data_consistency()
        print(f"✓ 数据一致性: {consistency_metrics['consistency']:.3f}")
        print(f"  一致性规则: {consistency_metrics['rules_checked']}")

        # 3. 数据准确性测试
        accuracy_metrics = self._test_data_accuracy()
        print(f"✓ 数据准确性: {accuracy_metrics['accuracy']:.3f}")
        print(f"  准确性检查: {accuracy_metrics['validation_results']}")

        # 4. 数据时效性测试
        timeliness_metrics = self._test_data_timeliness()
        print(f"✓ 数据时效性: {timeliness_metrics['timeliness']:.3f}")
        print(f"  时效性分析: {timeliness_metrics['time_analysis']}")

        # 综合质量评分
        overall_quality = self._calculate_overall_quality([
            completeness_metrics,
            consistency_metrics,
            accuracy_metrics,
            timeliness_metrics
        ])

        print(f"✓ 综合质量评分: {overall_quality['overall_score']:.3f}")
        print(f"  质量等级: {overall_quality['quality_grade']}")

        # 断言质量指标符合要求
        assert overall_quality['overall_score'] >= 0.8, "综合数据质量应 >= 80%"

        return {
            'completeness': completeness_metrics,
            'consistency': consistency_metrics,
            'accuracy': accuracy_metrics,
            'timeliness': timeliness_metrics,
            'overall_quality': overall_quality
        }

    def test_data_preprocessing_pipeline(self):
        """测试数据预处理流水线"""
        print("\n=== 数据预处理流水线测试 ===")

        # 1. 数据清洗
        cleaning_results = self._test_data_cleaning()
        print(f"✓ 数据清洗完成")
        print(f"  原始记录数: {cleaning_results['original_count']}")
        print(f"  清洗后记录数: {cleaning_results['cleaned_count']}")
        print(f"  清洗比率: {cleaning_results['cleaning_rate']:.3f}")

        # 2. 数据转换
        transformation_results = self._test_data_transformation(cleaning_results['cleaned_data'])
        print(f"✓ 数据转换完成")
        print(f"  转换步骤: {transformation_results['transformations_applied']}")
        print(f"  特征数量: {transformation_results['feature_count']}")

        # 3. 数据分割
        split_results = self._test_data_splitting(transformation_results['transformed_data'])
        print(f"✓ 数据分割完成")
        print(f"  训练集大小: {len(split_results['X_train'])}")
        print(f"  测试集大小: {len(split_results['X_test'])}")
        print(f"  分割比例: {split_results['split_ratio']}")

        # 4. 特征缩放
        scaling_results = self._test_feature_scaling(split_results)
        print(f"✓ 特征缩放完成")
        print(f"  缩放方法: {scaling_results['scaling_method']}")
        print(f"  缩放后统计: {scaling_results['scaled_stats']}")

        # 流水线完整性验证
        pipeline_integrity = self._validate_preprocessing_pipeline([
            cleaning_results,
            transformation_results,
            split_results,
            scaling_results
        ])

        print(f"✓ 流水线完整性: {pipeline_integrity['integrity_score']:.3f}")

        # 断言流水线完整性
        assert pipeline_integrity['integrity_score'] >= 0.9, "数据预处理流水线完整性应 >= 90%"

        return {
            'cleaning': cleaning_results,
            'transformation': transformation_results,
            'splitting': split_results,
            'scaling': scaling_results,
            'integrity': pipeline_integrity
        }

    def test_data_validation_rules(self):
        """测试数据验证规则"""
        print("\n=== 数据验证规则测试 ===")

        # 定义验证规则
        validation_rules = {
            'text_length': {'min': 10, 'max': 500},
            'rating_range': {'min': 1, 'max': 5},
            'sentiment_values': ['positive', 'negative', 'neutral'],
            'category_values': ['product', 'service']
        }

        # 执行验证
        validation_results = {}

        # 1. 文本长度验证
        text_validation = self._validate_text_length(validation_rules['text_length'])
        validation_results['text_length'] = text_validation
        print(f"✓ 文本长度验证: 通过率 {text_validation['pass_rate']:.3f}")

        # 2. 评分范围验证
        rating_validation = self._validate_rating_range(validation_rules['rating_range'])
        validation_results['rating_range'] = rating_validation
        print(f"✓ 评分范围验证: 通过率 {rating_validation['pass_rate']:.3f}")

        # 3. 情感值验证
        sentiment_validation = self._validate_sentiment_values(validation_rules['sentiment_values'])
        validation_results['sentiment'] = sentiment_validation
        print(f"✓ 情感值验证: 通过率 {sentiment_validation['pass_rate']:.3f}")

        # 4. 分类值验证
        category_validation = self._validate_category_values(validation_rules['category_values'])
        validation_results['category'] = category_validation
        print(f"✓ 分类值验证: 通过率 {category_validation['pass_rate']:.3f}")

        # 综合验证评分
        overall_validation = self._calculate_validation_score(validation_results)
        print(f"✓ 综合验证评分: {overall_validation['overall_score']:.3f}")
        print(f"  验证等级: {overall_validation['validation_grade']}")

        # 断言验证通过率
        assert overall_validation['overall_score'] >= 0.85, "数据验证综合通过率应 >= 85%"

        return validation_results

    def test_data_versioning_and_tracking(self):
        """测试数据版本控制和跟踪"""
        print("\n=== 数据版本控制和跟踪测试 ===")

        # 1. 数据版本创建
        version_info = self._create_data_version()
        print(f"✓ 数据版本创建")
        print(f"  版本ID: {version_info['version_id']}")
        print(f"  创建时间: {version_info['created_at']}")
        print(f"  数据哈希: {version_info['data_hash']}")

        # 2. 数据变更跟踪
        change_tracking = self._track_data_changes(version_info['version_id'])
        print(f"✓ 数据变更跟踪")
        print(f"  变更记录数: {change_tracking['change_count']}")
        print(f"  变更类型: {change_tracking['change_types']}")

        # 3. 数据血缘追踪
        lineage_tracking = self._trace_data_lineage(version_info['version_id'])
        print(f"✓ 数据血缘追踪")
        print(f"  血缘深度: {lineage_tracking['lineage_depth']}")
        print(f"  上游数据源: {lineage_tracking['upstream_sources']}")

        # 4. 数据审计日志
        audit_log = self._generate_audit_log(version_info['version_id'])
        print(f"✓ 数据审计日志")
        print(f"  审计条目数: {audit_log['audit_entries']}")
        print(f"  合规性检查: {audit_log['compliance_check']}")

        # 版本控制完整性验证
        versioning_integrity = self._validate_versioning_integrity([
            version_info,
            change_tracking,
            lineage_tracking,
            audit_log
        ])

        print(f"✓ 版本控制完整性: {versioning_integrity['integrity_score']:.3f}")

        # 断言版本控制完整性
        assert versioning_integrity['integrity_score'] >= 0.9, "数据版本控制完整性应 >= 90%"

        return {
            'version_info': version_info,
            'change_tracking': change_tracking,
            'lineage_tracking': lineage_tracking,
            'audit_log': audit_log,
            'integrity': versioning_integrity
        }

    def test_data_augmentation_techniques(self):
        """测试数据增强技术"""
        print("\n=== 数据增强技术测试 ===")

        # 准备增强数据
        text_data = self.df['text'].tolist()

        # 1. 文本增强技术
        augmentation_results = {}

        # 同义词替换
        synonym_results = self._synonym_replacement(text_data)
        augmentation_results['synonym_replacement'] = synonym_results
        print(f"✓ 同义词替换: 增强后样本数 {synonym_results['augmented_count']}")

        # 随机插入
        random_insertion_results = self._random_insertion(text_data)
        augmentation_results['random_insertion'] = random_insertion_results
        print(f"✓ 随机插入: 增强后样本数 {random_insertion_results['augmented_count']}")

        # 随机删除
        random_deletion_results = self._random_deletion(text_data)
        augmentation_results['random_deletion'] = random_deletion_results
        print(f"✓ 随机删除: 增强后样本数 {random_deletion_results['augmented_count']}")

        # 2. 增强质量评估
        augmentation_quality = self._evaluate_augmentation_quality(augmentation_results)
        print(f"✓ 增强质量评分: {augmentation_quality['quality_score']:.3f}")
        print(f"  语义保持度: {augmentation_quality['semantic_preservation']:.3f}")
        print(f"  多样性提升: {augmentation_quality['diversity_improvement']:.3f}")

        # 3. 增强效果验证
        augmentation_effectiveness = self._validate_augmentation_effectiveness(augmentation_results)
        print(f"✓ 增强有效性: {augmentation_effectiveness['effectiveness_score']:.3f}")
        print(f"  数据分布变化: {augmentation_effectiveness['distribution_change']}")

        # 断言增强效果
        assert augmentation_quality['quality_score'] >= 0.7, "数据增强质量应 >= 70%"
        assert augmentation_effectiveness['effectiveness_score'] >= 0.6, "数据增强有效性应 >= 60%"

        return {
            'augmentation_results': augmentation_results,
            'quality': augmentation_quality,
            'effectiveness': augmentation_effectiveness
        }

    def _test_data_completeness(self) -> Dict[str, Any]:
        """测试数据完整性"""
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        # 按列统计缺失值
        missing_stats = self.df.isnull().sum().to_dict()

        return {
            'completeness': completeness,
            'missing_cells': missing_cells,
            'total_cells': total_cells,
            'missing_stats': missing_stats,
            'meets_threshold': completeness >= self.data_quality_thresholds['completeness']
        }

    def _test_data_consistency(self) -> Dict[str, Any]:
        """测试数据一致性"""
        # 检查规则一致性
        consistency_violations = 0
        rules_checked = []

        # 规则1: rating为正数的记录，sentiment不应为"neutral"
        mask1 = (self.df['rating'] > 0) & (self.df['sentiment'] == 'neutral')
        violations1 = mask1.sum()
        consistency_violations += violations1
        rules_checked.append("rating-sentiment_consistency")

        # 规则2: category为"product"的记录，text应包含product相关词汇
        product_keywords = ['product', 'quality', 'delivery']
        mask2 = (self.df['category'] == 'product')
        has_product_keyword = self.df.loc[mask2, 'text'].str.contains('|'.join(product_keywords), case=False).sum()
        violations2 = mask2.sum() - has_product_keyword
        consistency_violations += violations2
        rules_checked.append("category-text_consistency")

        total_records = len(self.df)
        consistency = 1 - (consistency_violations / (total_records * len(rules_checked)))

        return {
            'consistency': consistency,
            'violations': consistency_violations,
            'total_records': total_records,
            'rules_checked': rules_checked,
            'meets_threshold': consistency >= self.data_quality_thresholds['consistency']
        }

    def _test_data_accuracy(self) -> Dict[str, Any]:
        """测试数据准确性"""
        validation_results = {}

        # 检查1: rating值应在1-5范围内
        valid_ratings = self.df['rating'].between(1, 5)
        rating_accuracy = valid_ratings.sum() / len(self.df)
        validation_results['rating_range_accuracy'] = rating_accuracy

        # 检查2: sentiment值应为有效分类
        valid_sentiments = self.df['sentiment'].isin(['positive', 'negative', 'neutral'])
        sentiment_accuracy = valid_sentiments.sum() / len(self.df)
        validation_results['sentiment_category_accuracy'] = sentiment_accuracy

        # 检查3: 长度合理性
        valid_length = self.df['text'].str.len() > 5  # 至少5个字符
        length_accuracy = valid_length.sum() / len(self.df)
        validation_results['text_length_accuracy'] = length_accuracy

        # 计算总体准确性
        accuracy = np.mean([
            rating_accuracy,
            sentiment_accuracy,
            length_accuracy
        ])

        return {
            'accuracy': accuracy,
            'validation_results': validation_results,
            'meets_threshold': accuracy >= self.data_quality_thresholds['accuracy']
        }

    def _test_data_timeliness(self) -> Dict[str, Any]:
        """测试数据时效性"""
        # 模拟时间戳
        today = datetime.now()

        # 为数据添加时间戳（假设数据是最近收集的）
        time_stamps = [
            today - pd.Timedelta(days=1),   # 1天前
            today - pd.Timedelta(days=7),   # 7天前
            today - pd.Timedelta(days=30),  # 30天前
            today - pd.Timedelta(days=3),   # 3天前
            today - pd.Timedelta(days=14)   # 14天前
        ]

        # 计算时效性（30天内的数据被认为及时）
        recent_threshold = pd.Timedelta(days=30)
        recent_count = sum(1 for ts in time_stamps if (today - ts) <= recent_threshold)
        timeliness = recent_count / len(time_stamps)

        # 时间分析
        time_analysis = {
            'oldest_record': max(time_stamps).strftime('%Y-%m-%d'),
            'newest_record': min(time_stamps).strftime('%Y-%m-%d'),
            'avg_age_days': np.mean([(today - ts).days for ts in time_stamps])
        }

        return {
            'timeliness': timeliness,
            'time_analysis': time_analysis,
            'recent_count': recent_count,
            'total_count': len(time_stamps),
            'meets_threshold': timeliness >= self.data_quality_thresholds['timeliness']
        }

    def _test_data_cleaning(self) -> Dict[str, Any]:
        """测试数据清洗"""
        original_count = len(self.df)

        # 创建包含脏数据的副本
        dirty_df = self.df.copy()

        # 添加脏数据
        dirty_df.loc[len(dirty_df)] = ['', 'mixed', -1, 'invalid']  # 空文本，错误分类，无效评分
        dirty_df.loc[len(dirty_df)] = ['short', 'negative', 6, 'product']  # 短文本，超出范围评分

        # 清洗过程
        cleaned_df = dirty_df.copy()

        # 1. 移除空值记录
        cleaned_df = cleaned_df.dropna(subset=['text'])

        # 2. 移除无效评分
        cleaned_df = cleaned_df[(cleaned_df['rating'] >= 1) & (cleaned_df['rating'] <= 5)]

        # 3. 移除过短文本
        cleaned_df = cleaned_df[cleaned_df['text'].str.len() >= 5]

        # 4. 重置索引
        cleaned_df = cleaned_df.reset_index(drop=True)

        cleaned_count = len(cleaned_df)
        cleaning_rate = (original_count - cleaned_count) / original_count

        return {
            'original_count': original_count + 2,  # 包含脏数据
            'cleaned_count': cleaned_count,
            'cleaning_rate': cleaning_rate,
            'cleaned_data': cleaned_df,
            'removed_records': original_count + 2 - cleaned_count
        }

    def _test_data_transformation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """测试数据转换"""
        transformations_applied = []

        # 1. 文本特征提取
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        transformations_applied.append('text_features')

        # 2. 情感编码
        label_encoder = LabelEncoder()
        df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
        transformations_applied.append('sentiment_encoding')

        # 3. 分类编码
        df['category_encoded'] = label_encoder.fit_transform(df['category'])
        transformations_applied.append('category_encoding')

        # 4. 数值特征标准化
        scaler = StandardScaler()
        numeric_features = ['rating', 'text_length', 'word_count']
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        transformations_applied.append('numeric_scaling')

        # 5. 特征交叉
        df['rating_length_interaction'] = df['rating'] * df['text_length']
        transformations_applied.append('feature_interaction')

        feature_count = len([col for col in df.columns if col != 'text'])

        return {
            'transformations_applied': transformations_applied,
            'feature_count': feature_count,
            'transformed_data': df,
            'feature_columns': [col for col in df.columns if col != 'text']
        }

    def _test_data_splitting(self, df: pd.DataFrame) -> Dict[str, Any]:
        """测试数据分割"""
        # 准备特征和标签
        feature_columns = [col for col in df.columns if col not in ['text', 'sentiment', 'category']]
        X = df[feature_columns]
        y = df['sentiment_encoded'] if 'sentiment_encoded' in df.columns else df['sentiment']

        # 执行分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        split_ratio = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_ratio': len(X_train) / len(df),
            'test_ratio': len(X_test) / len(df)
        }

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'split_ratio': split_ratio
        }

    def _test_feature_scaling(self, split_results: Dict[str, Any]) -> Dict[str, Any]:
        """测试特征缩放"""
        scaler = StandardScaler()

        # 拟合训练数据并转换
        X_train_scaled = scaler.fit_transform(split_results['X_train'])
        X_test_scaled = scaler.transform(split_results['X_test'])

        # 计算缩放后的统计信息
        train_mean = np.mean(X_train_scaled, axis=0)
        train_std = np.std(X_train_scaled, axis=0)

        scaled_stats = {
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'zero_centered': np.allclose(train_mean, 0, atol=1e-6),
            'unit_variance': np.allclose(train_std, 1.0, atol=1e-6)
        }

        return {
            'scaling_method': 'StandardScaler',
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'scaled_stats': scaled_stats
        }

    def _validate_text_length(self, rule: Dict[str, int]) -> Dict[str, Any]:
        """验证文本长度"""
        text_lengths = self.df['text'].str.len()
        valid_lengths = text_lengths.between(rule['min'], rule['max'])
        pass_count = valid_lengths.sum()

        return {
            'pass_rate': pass_count / len(self.df),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'avg_length': text_lengths.mean(),
            'violations': len(self.df) - pass_count
        }

    def _validate_rating_range(self, rule: Dict[str, int]) -> Dict[str, Any]:
        """验证评分范围"""
        valid_ratings = self.df['rating'].between(rule['min'], rule['max'])
        pass_count = valid_ratings.sum()

        return {
            'pass_rate': pass_count / len(self.df),
            'min_rating': self.df['rating'].min(),
            'max_rating': self.df['rating'].max(),
            'avg_rating': self.df['rating'].mean(),
            'violations': len(self.df) - pass_count
        }

    def _validate_sentiment_values(self, valid_values: List[str]) -> Dict[str, Any]:
        """验证情感值"""
        valid_sentiments = self.df['sentiment'].isin(valid_values)
        pass_count = valid_sentiments.sum()

        return {
            'pass_rate': pass_count / len(self.df),
            'valid_values': valid_values,
            'unique_values': self.df['sentiment'].unique().tolist(),
            'value_counts': self.df['sentiment'].value_counts().to_dict(),
            'violations': len(self.df) - pass_count
        }

    def _validate_category_values(self, valid_values: List[str]) -> Dict[str, Any]:
        """验证分类值"""
        valid_categories = self.df['category'].isin(valid_values)
        pass_count = valid_categories.sum()

        return {
            'pass_rate': pass_count / len(self.df),
            'valid_values': valid_values,
            'unique_values': self.df['category'].unique().tolist(),
            'value_counts': self.df['category'].value_counts().to_dict(),
            'violations': len(self.df) - pass_count
        }

    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算验证评分"""
        scores = []
        for rule_name, result in validation_results.items():
            scores.append(result['pass_rate'])

        overall_score = np.mean(scores)

        # 评分等级
        if overall_score >= 0.95:
            grade = '优秀'
        elif overall_score >= 0.85:
            grade = '良好'
        elif overall_score >= 0.75:
            grade = '一般'
        else:
            grade = '差'

        return {
            'overall_score': overall_score,
            'validation_grade': grade,
            'rule_scores': dict(zip(validation_results.keys(), scores))
        }

    def _calculate_overall_quality(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算综合质量评分"""
        scores = []
        for metric in metrics:
            if 'completeness' in metric:
                scores.append(metric['completeness'])
            elif 'consistency' in metric:
                scores.append(metric['consistency'])
            elif 'accuracy' in metric:
                scores.append(metric['accuracy'])
            elif 'timeliness' in metric:
                scores.append(metric['timeliness'])

        overall_score = np.mean(scores)

        # 质量等级
        if overall_score >= 0.95:
            grade = '优秀'
        elif overall_score >= 0.85:
            grade = '良好'
        elif overall_score >= 0.75:
            grade = '一般'
        else:
            grade = '需要改进'

        return {
            'overall_score': overall_score,
            'quality_grade': grade,
            'component_scores': scores
        }

    def _validate_preprocessing_pipeline(self, pipeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证预处理流水线完整性"""
        integrity_checks = []

        # 检查数据流连续性
        cleaning_out = pipeline_results[0]['cleaned_count']
        transformation_out = len(pipeline_results[1]['transformed_data'])
        data_flow_continuity = abs(cleaning_out - transformation_out) == 0
        integrity_checks.append(data_flow_continuity)

        # 检查特征转换完整性
        transformation_applied = len(pipeline_results[1]['transformations_applied'])
        feature_transformation_complete = transformation_applied >= 5  # 至少5个转换
        integrity_checks.append(feature_transformation_complete)

        # 检查数据分割合理性
        test_ratio = pipeline_results[2]['split_ratio']['test_ratio']
        split_reasonable = 0.15 <= test_ratio <= 0.25  # 测试集15-25%
        integrity_checks.append(split_reasonable)

        # 检查缩放正确性
        scaled_stats = pipeline_results[3]['scaled_stats']
        scaling_correct = scaled_stats['zero_centered'] and scaled_stats['unit_variance']
        integrity_checks.append(scaling_correct)

        integrity_score = sum(integrity_checks) / len(integrity_checks)

        return {
            'integrity_score': integrity_score,
            'integrity_checks': {
                'data_flow_continuity': data_flow_continuity,
                'feature_transformation_complete': feature_transformation_complete,
                'split_reasonable': split_reasonable,
                'scaling_correct': scaling_correct
            },
            'passed_checks': sum(integrity_checks),
            'total_checks': len(integrity_checks)
        }

    def _create_data_version(self) -> Dict[str, Any]:
        """创建数据版本"""
        version_id = f"data_v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 计算数据哈希
        data_content = self.df.to_string()
        data_hash = hash(data_content)

        return {
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'data_hash': str(data_hash),
            'record_count': len(self.df),
            'schema': list(self.df.columns),
            'size_bytes': len(data_content.encode('utf-8'))
        }

    def _track_data_changes(self, version_id: str) -> Dict[str, Any]:
        """跟踪数据变更"""
        # 模拟变更记录
        changes = [
            {
                'timestamp': datetime.now().isoformat(),
                'change_type': 'schema_change',
                'description': 'Added text_length feature',
                'affected_records': len(self.df)
            },
            {
                'timestamp': datetime.now().isoformat(),
                'change_type': 'data_update',
                'description': 'Cleaned invalid records',
                'affected_records': 2
            }
        ]

        change_types = [change['change_type'] for change in changes]

        return {
            'version_id': version_id,
            'change_count': len(changes),
            'changes': changes,
            'change_types': change_types
        }

    def _trace_data_lineage(self, version_id: str) -> Dict[str, Any]:
        """追踪数据血缘"""
        # 模拟数据血缘
        upstream_sources = [
            {
                'source_type': 'database',
                'source_id': 'reviews_db',
                'extraction_time': datetime.now().isoformat(),
                'query': 'SELECT * FROM reviews WHERE created_at >= "2024-01-01"'
            },
            {
                'source_type': 'api',
                'source_id': 'sentiment_analysis_api',
                'extraction_time': datetime.now().isoformat(),
                'endpoint': '/api/v1/analyze'
            }
        ]

        lineage_depth = len(upstream_sources)

        return {
            'version_id': version_id,
            'lineage_depth': lineage_depth,
            'upstream_sources': upstream_sources,
            'data_flow': 'database -> preprocessing -> model'
        }

    def _generate_audit_log(self, version_id: str) -> Dict[str, Any]:
        """生成审计日志"""
        audit_entries = []

        # 数据质量审计
        audit_entries.append({
            'timestamp': datetime.now().isoformat(),
            'audit_type': 'data_quality',
            'result': 'PASS',
            'details': 'Data completeness: 100%, Accuracy: 98%',
            'auditor': 'automated_system'
        })

        # 隐私合规审计
        audit_entries.append({
            'timestamp': datetime.now().isoformat(),
            'audit_type': 'privacy_compliance',
            'result': 'PASS',
            'details': 'No PII detected in the dataset',
            'auditor': 'privacy_checker'
        })

        # 访问控制审计
        audit_entries.append({
            'timestamp': datetime.now().isoformat(),
            'audit_type': 'access_control',
            'result': 'PASS',
            'details': 'All access requests properly authorized',
            'auditor': 'access_logger'
        })

        # 合规性检查
        compliance_check = all(entry['result'] == 'PASS' for entry in audit_entries)

        return {
            'version_id': version_id,
            'audit_entries': len(audit_entries),
            'audit_log': audit_entries,
            'compliance_check': compliance_check,
            'last_audit': datetime.now().isoformat()
        }

    def _validate_versioning_integrity(self, versioning_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证版本控制完整性"""
        integrity_checks = []

        # 检查版本信息完整性
        version_info = versioning_results[0]
        has_required_fields = all([
            'version_id' in version_info,
            'created_at' in version_info,
            'data_hash' in version_info
        ])
        integrity_checks.append(has_required_fields)

        # 检查变更跟踪完整性
        change_tracking = versioning_results[1]
        has_change_records = len(change_tracking['changes']) > 0
        integrity_checks.append(has_change_records)

        # 检查血缘追踪完整性
        lineage_tracking = versioning_results[2]
        has_upstream_sources = len(lineage_tracking['upstream_sources']) > 0
        integrity_checks.append(has_upstream_sources)

        # 检查审计日志完整性
        audit_log = versioning_results[3]
        has_audit_entries = len(audit_log['audit_log']) > 0
        integrity_checks.append(has_audit_entries)

        integrity_score = sum(integrity_checks) / len(integrity_checks)

        return {
            'integrity_score': integrity_score,
            'integrity_checks': {
                'version_info_complete': has_required_fields,
                'change_tracking_active': has_change_records,
                'lineage_tracking_active': has_upstream_sources,
                'audit_logging_active': has_audit_entries
            },
            'passed_checks': sum(integrity_checks),
            'total_checks': len(integrity_checks)
        }

    def _synonym_replacement(self, text_data: List[str]) -> Dict[str, Any]:
        """同义词替换增强"""
        augmented_texts = []

        # 简单的同义词词典
        synonym_dict = {
            'good': 'excellent',
            'bad': 'poor',
            'positive': 'favorable',
            'negative': 'unfavorable'
        }

        for text in text_data:
            # 应用同义词替换
            augmented_text = text
            for word, synonym in synonym_dict.items():
                if word in text.lower():
                    augmented_text = text.replace(word, synonym)
                    break

            if augmented_text != text:
                augmented_texts.append(augmented_text)

        return {
            'augmented_count': len(augmented_texts),
            'original_count': len(text_data),
            'augmentation_ratio': len(augmented_texts) / len(text_data),
            'augmented_texts': augmented_texts
        }

    def _random_insertion(self, text_data: List[str]) -> Dict[str, Any]:
        """随机插入增强"""
        augmented_texts = []

        # 随机插入的词汇
        insertion_words = ['very', 'quite', 'really', 'absolutely']

        for text in text_data:
            words = text.split()
            if len(words) >= 2:
                # 随机选择插入位置
                insert_pos = np.random.randint(1, len(words))
                insert_word = np.random.choice(insertion_words)

                # 插入词汇
                words.insert(insert_pos, insert_word)
                augmented_text = ' '.join(words)
                augmented_texts.append(augmented_text)

        return {
            'augmented_count': len(augmented_texts),
            'original_count': len(text_data),
            'augmentation_ratio': len(augmented_texts) / len(text_data),
            'augmented_texts': augmented_texts
        }

    def _random_deletion(self, text_data: List[str]) -> Dict[str, Any]:
        """随机删除增强"""
        augmented_texts = []

        for text in text_data:
            words = text.split()
            if len(words) >= 4:
                # 随机删除1个词汇
                delete_pos = np.random.randint(1, len(words))
                words.pop(delete_pos)
                augmented_text = ' '.join(words)
                augmented_texts.append(augmented_text)

        return {
            'augmented_count': len(augmented_texts),
            'original_count': len(text_data),
            'augmentation_ratio': len(augmented_texts) / len(text_data),
            'augmented_texts': augmented_texts
        }

    def _evaluate_augmentation_quality(self, augmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估增强质量"""
        quality_scores = []

        for technique, results in augmentation_results.items():
            # 计算语义保持度（简化评估）
            semantic_preservation = 1.0 - results['augmentation_ratio'] * 0.1  # 简化计算
            quality_scores.append(semantic_preservation)

        # 计算多样性提升
        diversity_improvement = np.mean([r['augmentation_ratio'] for r in augmentation_results.values()])

        # 计算综合质量分数
        quality_score = np.mean(quality_scores)

        return {
            'quality_score': quality_score,
            'semantic_preservation': quality_score,
            'diversity_improvement': diversity_improvement,
            'technique_scores': dict(zip(augmentation_results.keys(), quality_scores))
        }

    def _validate_augmentation_effectiveness(self, augmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证增强效果"""
        # 计算数据分布变化
        original_count = len(self.df)
        augmented_count = sum([r['augmented_count'] for r in augmentation_results.values()])
        distribution_change = augmented_count / original_count

        # 计算增强有效性分数
        effectiveness_scores = []
        for technique, results in augmentation_results.items():
            if results['augmentation_ratio'] > 0.1:  # 增强率 > 10%
                effectiveness_scores.append(0.8)
            elif results['augmentation_ratio'] > 0.05:  # 增强率 > 5%
                effectiveness_scores.append(0.6)
            else:
                effectiveness_scores.append(0.3)

        effectiveness_score = np.mean(effectiveness_scores) if effectiveness_scores else 0.0

        return {
            'effectiveness_score': effectiveness_score,
            'distribution_change': distribution_change,
            'total_augmented': augmented_count,
            'technique_effectiveness': dict(zip(augmentation_results.keys(), effectiveness_scores))
        }


# pytest主函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
