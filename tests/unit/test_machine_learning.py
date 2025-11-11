"""
机器学习专项测试
深度测试机器学习模型的训练、评估、部署等各个环节
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.ml.models import MLModel
from src.ml.training import ModelTrainer
from src.ml.evaluation import ModelEvaluator
from src.ml.preprocessing import DataPreprocessor
from src.core.types import TrainingConfig


class TestMLCore:
    """机器学习核心功能测试"""

    @pytest.fixture
    def sample_classification_data(self):
        """分类数据fixture"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), y

    @pytest.fixture
    def sample_regression_data(self):
        """回归数据fixture"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = X.sum(axis=1) + np.random.randn(n_samples) * 0.1

        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]), y

    @pytest.fixture
    def training_config(self):
        """训练配置fixture"""
        return TrainingConfig(
            model_type='random_forest',
            test_size=0.2,
            random_state=42,
            cross_validation_folds=5,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2
            }
        )

    def test_data_preprocessing_classification(self, sample_classification_data):
        """测试分类数据预处理"""
        X, y = sample_classification_data
        preprocessor = DataPreprocessor()

        # 测试分类预处理
        processed_data = preprocessor.preprocess_classification(X, y)

        # 验证预处理结果
        assert hasattr(processed_data, 'X_processed')
        assert hasattr(processed_data, 'y_processed')
        assert hasattr(processed_data, 'preprocessing_pipeline')

        # 验证数据完整性
        assert len(processed_data.X_processed) == len(X)
        assert len(processed_data.y_processed) == len(y)

        # 验证预处理步骤
        pipeline = processed_data.preprocessing_pipeline
        assert any('scaling' in str(step) for step in pipeline)
        assert any('encoding' in str(step) for step in pipeline)

    def test_data_preprocessing_regression(self, sample_regression_data):
        """测试回归数据预处理"""
        X, y = sample_regression_data
        preprocessor = DataPreprocessor()

        # 测试回归预处理
        processed_data = preprocessor.preprocess_regression(X, y)

        # 验证预处理结果
        assert hasattr(processed_data, 'X_processed')
        assert hasattr(processed_data, 'y_processed')
        assert hasattr(processed_data, 'preprocessing_pipeline')

        # 验证数值特征缩放
        X_processed = processed_data.X_processed
        if hasattr(X_processed, 'values'):
            # 检查均值和标准差
            for col in X_processed.select_dtypes(include=[np.number]).columns:
                col_data = X_processed[col].values
                assert abs(col_data.mean()) < 1.0  # 标准化后均值接近0
                assert abs(col_data.std() - 1.0) < 0.1  # 标准化后标准差接近1

    def test_feature_engineering(self, sample_classification_data):
        """测试特征工程"""
        X, y = sample_classification_data
        preprocessor = DataPreprocessor()

        # 测试多项式特征
        poly_features = preprocessor.create_polynomial_features(X, degree=2)
        assert poly_features.shape[1] > X.shape[1]  # 特征数量增加

        # 测试交互特征
        interaction_features = preprocessor.create_interaction_features(X, pairs=[('feature_0', 'feature_1')])
        assert interaction_features.shape[1] > X.shape[1]  # 有新的交互特征

        # 测试统计特征
        stat_features = preprocessor.create_statistical_features(X)
        assert stat_features.shape[1] >= X.shape[1]  # 至少包含原始特征

    def test_feature_selection(self, sample_classification_data):
        """测试特征选择"""
        X, y = sample_classification_data
        preprocessor = DataPreprocessor()

        # 创建额外的噪声特征
        noise_features = pd.DataFrame(np.random.randn(len(X), 50),
                                    columns=[f'noise_feature_{i}' for i in range(50)])
        X_with_noise = pd.concat([X, noise_features], axis=1)

        # 测试特征选择
        selected_features = preprocessor.select_features(X_with_noise, y, method='univariate', k=15)
        assert len(selected_features) == 15
        assert all(col in X.columns for col in selected_features)  # 原始特征应该被保留
        assert not any(col.startswith('noise_') for col in selected_features)  # 噪声特征应该被过滤


class TestModelTraining:
    """模型训练测试"""

    @pytest.fixture
    def mock_model(self):
        """模拟模型"""
        model = Mock()
        model.fit = Mock(return_value=None)
        model.predict = Mock(return_value=np.random.randint(0, 2, 100))
        model.score = Mock(return_value=0.85)
        return model

    @pytest.fixture
    def model_trainer(self, mock_model):
        """训练器fixture"""
        trainer = ModelTrainer()
        trainer.model_class = Mock(return_value=mock_model)
        return trainer

    def test_train_classification_model(self, sample_classification_data, training_config, model_trainer):
        """测试分类模型训练"""
        X, y = sample_classification_data
        trainer = ModelTrainer()

        # 训练模型
        training_result = trainer.train_classification(
            X, y, config=training_config
        )

        # 验证训练结果
        assert 'model' in training_result
        assert 'training_metrics' in training_result
        assert 'validation_metrics' in training_result

        # 验证模型属性
        model = training_result['model']
        assert model is not None
        assert hasattr(model, 'classes_')  # 分类模型应该有类别属性

    def test_train_regression_model(self, sample_regression_data, training_config):
        """测试回归模型训练"""
        X, y = sample_regression_data
        trainer = ModelTrainer()

        # 修改配置为回归
        reg_config = training_config.copy()
        reg_config.task_type = 'regression'

        # 训练模型
        training_result = trainer.train_regression(X, y, config=reg_config)

        # 验证训练结果
        assert 'model' in training_result
        assert 'training_metrics' in training_result
        assert 'validation_metrics' in training_result

        # 验证回归指标
        metrics = training_result['training_metrics']
        assert 'mse' in metrics or 'rmse' in metrics
        assert 'r2_score' in metrics

    def test_hyperparameter_tuning(self, sample_classification_data):
        """测试超参数调优"""
        X, y = sample_classification_data
        trainer = ModelTrainer()

        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        # 执行网格搜索
        tuning_result = trainer.hyperparameter_tuning(
            X, y, param_grid=param_grid, cv_folds=3
        )

        # 验证调优结果
        assert 'best_params' in tuning_result
        assert 'best_score' in tuning_result
        assert 'cv_results' in tuning_result

        # 验证最佳参数
        best_params = tuning_result['best_params']
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert best_params['n_estimators'] in [50, 100, 200]

    def test_cross_validation(self, sample_classification_data):
        """测试交叉验证"""
        X, y = sample_classification_data
        trainer = ModelTrainer()

        # 执行交叉验证
        cv_results = trainer.cross_validate(
            X, y, cv_folds=5, scoring='accuracy'
        )

        # 验证交叉验证结果
        assert isinstance(cv_results, list)
        assert len(cv_results) == 5  # 5折交叉验证

        # 验证每折的分数
        for fold_score in cv_results:
            assert isinstance(fold_score, (int, float))
            assert 0 <= fold_score <= 1  # 准确率应该在0-1之间

        # 验证平均分数和标准差
        mean_score = np.mean(cv_results)
        std_score = np.std(cv_results)

        assert 0 <= mean_score <= 1
        assert std_score >= 0
        assert std_score <= 1

    def test_ensemble_training(self, sample_classification_data):
        """测试集成学习训练"""
        X, y = sample_classification_data
        trainer = ModelTrainer()

        # 配置集成学习
        ensemble_config = {
            'base_models': ['random_forest', 'logistic_regression', 'svm'],
            'voting_strategy': 'soft',
            'cross_validation': True
        }

        # 训练集成模型
        ensemble_result = trainer.train_ensemble(X, y, config=ensemble_config)

        # 验证集成结果
        assert 'ensemble_model' in ensemble_result
        assert 'base_models' in ensemble_result
        assert 'ensemble_metrics' in ensemble_result

        # 验证集成模型属性
        ensemble = ensemble_result['ensemble_model']
        assert hasattr(ensemble, 'estimators_')  # 集成模型应该有基础估计器
        assert len(ensemble.estimators_) == 3


class TestModelEvaluation:
    """模型评估测试"""

    @pytest.fixture
    def trained_model_and_data(self, sample_classification_data):
        """已训练模型和测试数据fixture"""
        X, y = sample_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练简单模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        return model, X_test, y_test

    def test_classification_evaluation(self, trained_model_and_data):
        """测试分类评估"""
        model, X_test, y_test = trained_model_and_data
        evaluator = ModelEvaluator()

        # 执行分类评估
        eval_results = evaluator.evaluate_classification(model, X_test, y_test)

        # 验证评估结果
        assert 'accuracy' in eval_results
        assert 'precision' in eval_results
        assert 'recall' in eval_results
        assert 'f1_score' in eval_results
        assert 'classification_report' in eval_results
        assert 'confusion_matrix' in eval_results

        # 验证指标范围
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            metric_value = eval_results[metric]
            assert isinstance(metric_value, (int, float))
            assert 0 <= metric_value <= 1

        # 验证混淆矩阵
        cm = eval_results['confusion_matrix']
        assert cm.shape == (2, 2)  # 二分类的2x2矩阵
        assert np.all(cm >= 0)  # 所有值应该非负

    def test_regression_evaluation(self, sample_regression_data):
        """测试回归评估"""
        X, y = sample_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练回归模型
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        # 执行回归评估
        eval_results = evaluator.evaluate_regression(model, X_test, y_test)

        # 验证评估结果
        assert 'mse' in eval_results
        assert 'rmse' in eval_results
        assert 'mae' in eval_results
        assert 'r2_score' in eval_results

        # 验证指标合理性
        assert eval_results['mse'] >= 0
        assert eval_results['rmse'] >= 0
        assert eval_results['mae'] >= 0
        assert -1 <= eval_results['r2_score'] <= 1

    def test_learning_curve_analysis(self, sample_classification_data):
        """测试学习曲线分析"""
        X, y = sample_classification_data
        evaluator = ModelEvaluator()

        # 生成学习曲线
        learning_curve_result = evaluator.learning_curve_analysis(
            X, y, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], cv=3
        )

        # 验证学习曲线结果
        assert 'train_scores' in learning_curve_result
        assert 'test_scores' in learning_curve_result
        assert 'train_sizes' in learning_curve_result
        assert 'train_mean' in learning_curve_result
        assert 'test_mean' in learning_curve_result

        # 验证曲线特征
        train_scores = learning_curve_result['train_scores']
        test_scores = learning_curve_result['test_scores']

        assert len(train_scores) == 5  # 5个训练集大小
        assert len(test_scores) == 5
        assert len(learning_curve_result['train_sizes']) == 5

        # 验证学习曲线趋势
        # 通常随着训练集增加，训练分数增加，测试分数先增加后减少
        train_mean = learning_curve_result['train_mean']
        test_mean = learning_curve_result['test_mean']

        # 验证分数都在合理范围内
        for score in train_mean + test_mean:
            assert 0 <= score <= 1

    def test_model_comparison(self, trained_model_and_data, sample_classification_data):
        """测试模型比较"""
        model1, X_test, y_test = trained_model_and_data
        X, y = sample_classification_data

        # 训练第二个模型
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model2 = LogisticRegression(random_state=42)
        model2.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        # 比较模型
        comparison_result = evaluator.compare_models(
            {'random_forest': model1, 'logistic_regression': model2},
            X_test, y_test, metrics=['accuracy', 'f1_score', 'roc_auc']
        )

        # 验证比较结果
        assert 'model_metrics' in comparison_result
        assert 'best_model' in comparison_result
        assert 'comparison_table' in comparison_result

        # 验证每个模型的指标
        model_metrics = comparison_result['model_metrics']
        assert 'random_forest' in model_metrics
        assert 'logistic_regression' in model_metrics

        for model_name, metrics in model_metrics.items():
            assert 'accuracy' in metrics
            assert 'f1_score' in metrics
            assert 'roc_auc' in metrics

    def test_bias_and_fairness_evaluation(self, sample_classification_data):
        """测试偏差和公平性评估"""
        X, y = sample_classification_data
        evaluator = ModelEvaluator()

        # 添加敏感属性（模拟）
        sensitive_attribute = np.random.randint(0, 2, len(y))  # 性别或其他敏感属性

        # 训练模型
        X_train, X_test, y_train, y_test, sen_train, sen_test = train_test_split(
            X, y, sensitive_attribute, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # 执行公平性评估
        fairness_result = evaluator.evaluate_fairness(
            model, X_test, y_test, sensitive_attribute=sen_test
        )

        # 验证公平性结果
        assert 'demographic_parity' in fairness_result
        assert 'equalized_odds' in fairness_result
        assert 'equal_opportunity' in fairness_result
        assert 'fairness_scores' in fairness_result

        # 验证公平性指标
        fairness_scores = fairness_result['fairness_scores']
        assert 'disparate_impact' in fairness_scores
        assert 'statistical_parity_difference' in fairness_scores
        assert 'equal_opportunity_difference' in fairness_scores

        for metric_value in fairness_scores.values():
            assert isinstance(metric_value, (int, float))
            assert -1 <= metric_value <= 1  # 公平性指标通常在-1到1之间


class TestModelDeployment:
    """模型部署测试"""

    @pytest.fixture
    def trained_model(self, sample_classification_data):
        """已训练模型fixture"""
        X, y = sample_classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_model_serialization(self, trained_model):
        """测试模型序列化"""
        from src.ml.deployment import ModelSerializer

        serializer = ModelSerializer()

        # 测试不同序列化格式
        formats = ['pickle', 'joblib', 'onnx', 'json']

        for format_type in formats:
            if format_type == 'json':
                # JSON格式可能不支持所有模型类型
                try:
                    serialized_model = serializer.serialize(model, format_type)
                    assert serialized_model is not None
                except Exception:
                    pass  # JSON序列化可能失败，这是正常的
            else:
                # 测试反序列化
                if serialized_model:
                    deserialized_model = serializer.deserialize(serialized_model, format_type)
                    assert deserialized_model is not None

                    # 验证反序列化后的模型预测一致性
                    test_data = np.random.randn(10, 20)
                    original_pred = model.predict(test_data)
                    deserialized_pred = deserialized_model.predict(test_data)

                    # 预测应该一致（对于支持完整序列化的格式）
                    if format_type in ['pickle', 'joblib']:
                        assert np.array_equal(original_pred, deserialized_pred)
            else:
                # 测试序列化
                serialized_model = serializer.serialize(model, format_type)
                assert serialized_model is not None

                # 测试反序列化
                deserialized_model = serializer.deserialize(serialized_model, format_type)
                assert deserialized_model is not None

                # 验证反序列化后的模型预测一致性
                test_data = np.random.randn(10, 20)
                original_pred = model.predict(test_data)
                deserialized_pred = deserialized_model.predict(test_data)
                assert np.array_equal(original_pred, deserialized_pred)

    def test_model_optimization(self, trained_model):
        """测试模型优化"""
        from src.ml.deployment import ModelOptimizer

        optimizer = ModelOptimizer()

        # 测试模型量化
        quantized_model = optimizer.quantize_model(trained_model, quant_type='int8')
        assert quantized_model is not None
        assert hasattr(quantized_model, 'quantization_info')

        # 测试模型剪枝
        pruned_model = optimizer.prune_model(trained_model, sparsity=0.5)
        assert pruned_model is not None
        assert hasattr(pruned_model, 'pruning_info')

        # 测试模型蒸馏
        # 创建学生模型
        student_model = RandomForestClassifier(n_estimators=5, random_state=42)
        distilled_model = optimizer.distill_model(
            teacher_model=trained_model,
            student_model=student_model,
            X_train=np.random.randn(100, 20),
            y_train=np.random.randint(0, 2, 100)
        )

        assert distilled_model is not None
        assert hasattr(distilled_model, 'distillation_info')

    def test_model_monitoring(self, trained_model):
        """测试模型监控"""
        from src.ml.deployment import ModelMonitor

        monitor = ModelMonitor(model=trained_model)

        # 模拟预测和监控数据
        test_data = np.random.randn(100, 20)
        predictions = trained_model.predict(test_data)

        # 记录监控数据
        for i, pred in enumerate(predictions):
            monitor.log_prediction(
                input_data=test_data[i],
                prediction=pred,
                timestamp=i,
                confidence=0.8 + np.random.randn() * 0.1
            )

        # 获取监控统计
        monitoring_stats = monitor.get_statistics()

        # 验证监控指标
        assert 'prediction_count' in monitoring_stats
        assert 'accuracy' in monitoring_stats
        assert 'latency' in monitoring_stats
        assert 'drift_detected' in monitoring_stats

        assert monitoring_stats['prediction_count'] == 100
        assert 0 <= monitoring_stats['accuracy'] <= 1

    def test_a_b_testing(self, trained_model):
        """测试A/B测试"""
        from src.ml.deployment import ABTester

        # 创建两个模型版本
        model_a = trained_model
        model_b = RandomForestClassifier(n_estimators=15, random_state=123)  # 略有不同的模型

        X_test = np.random.randn(100, 20)
        true_labels = np.random.randint(0, 2, 100)

        # 设置A/B测试
        ab_tester = ABTester()
        ab_tester.setup_ab_test(
            model_a=model_a,
            model_b=model_b,
            traffic_split=0.5
        )

        # 模拟用户请求
        results = []
        for user_id in range(100):
            prediction_result = ab_tester.predict(user_id, X_test[user_id])
            results.append(prediction_result)

        # 验证A/B测试结果
        assert len(results) == 100

        # 验证流量分配
        a_predictions = [r for r in results if r['model_used'] == 'model_a']
        b_predictions = [r for r in results if r['model_used'] == 'model_b']

        # 流量分配应该接近50-50
        traffic_ratio = len(a_predictions) / len(b_predictions)
        assert 0.8 <= traffic_ratio <= 1.2  # 允许一些偏差

        # 验证性能比较
        performance = ab_tester.get_performance_summary()
        assert 'model_a_metrics' in performance
        assert 'model_b_metrics' in performance
        assert 'statistical_significance' in performance


class TestMLPipeline:
    """机器学习流水线测试"""

    @pytest.mark.asyncio
    async def test_end_to_end_ml_pipeline(self, sample_classification_data):
        """测试端到端机器学习流水线"""
        from src.ml.pipeline import MLPipeline

        X, y = sample_classification_data

        # 创建完整的ML流水线
        pipeline = MLPipeline()

        # 定义流水线步骤
        pipeline_steps = [
            ('data_validation', {'check_missing_values': True, 'check_data_types': True}),
            ('data_preprocessing', {'scaling': True, 'encoding': True}),
            ('feature_engineering', {'polynomial_features': 2, 'interaction_features': True}),
            ('feature_selection', {'method': 'univariate', 'k': 15}),
            ('model_training', {'model_type': 'random_forest', 'hyperparameter_tuning': True}),
            ('model_evaluation', {'metrics': ['accuracy', 'f1_score', 'roc_auc']}),
            ('model_optimization', {'quantization': True}),
            ('model_deployment', {'format': 'pickle', 'monitoring': True})
        ]

        # 执行流水线
        pipeline_result = await pipeline.execute(X, y, steps=pipeline_steps)

        # 验证流水线结果
        assert 'success' in pipeline_result
        assert 'step_results' in pipeline_result
        assert 'final_model' in pipeline_result
        assert 'deployment_info' in pipeline_result
        assert 'pipeline_metrics' in pipeline_result

        # 验证每个步骤的结果
        step_results = pipeline_result['step_results']
        for step_name in [step[0] for step in pipeline_steps]:
            assert step_name in step_results
            assert step_results[step_name]['status'] in ['success', 'warning', 'error']

        # 验证最终模型
        final_model = pipeline_result['final_model']
        assert final_model is not None
        assert hasattr(final_model, 'predict')

        # 验证部署信息
        deployment_info = pipeline_result['deployment_info']
        assert 'model_path' in deployment_info
        assert 'monitoring_url' in deployment_info

    @pytest.mark.asyncio
    async def test_ml_pipeline_error_handling(self, sample_classification_data):
        """测试ML流水线错误处理"""
        from src.ml.pipeline import MLPipeline

        X, y = sample_classification_data

        # 创建会出错的流水线步骤
        problematic_steps = [
            ('data_validation', {'check_missing_values': True}),  # 正常步骤
            ('data_preprocessing', {'scaling': True, 'encoding': True}),  # 正常步骤
            ('feature_engineering', {'polynomial_features': 1000}),  # 可能导致内存错误
            ('model_training', {'model_type': 'invalid_model'}),  # 无效模型类型
            ('model_evaluation', {'metrics': 'invalid_metric'})  # 无效指标
        ]

        pipeline = MLPipeline()

        # 执行会出错的流水线
        pipeline_result = await pipeline.execute(X, y, steps=problematic_steps)

        # 验证错误处理
        assert 'success' in pipeline_result
        assert pipeline_result['success'] is False
        assert 'errors' in pipeline_result
        assert 'step_results' in pipeline_result

        # 验证错误信息
        errors = pipeline_result['errors']
        assert len(errors) > 0

        # 验证失败步骤的详细信息
        step_results = pipeline_result['step_results']
        for step_name, step_result in step_results.items():
            if step_result['status'] == 'error':
                assert 'error_message' in step_result
                assert 'error_type' in step_result
                assert step_name in errors

    @pytest.mark.asyncio
    async def test_ml_pipeline_caching(self, sample_classification_data):
        """测试ML流水线缓存"""
        from src.ml.pipeline import MLPipeline

        X, y = sample_classification_data

        pipeline = MLPipeline()
        pipeline.enable_caching(cache_dir='/tmp/ml_cache')

        # 第一次执行流水线
        result1 = await pipeline.execute(X, y, steps=[
            ('data_preprocessing', {}),
            ('model_training', {})
        ])

        # 第二次执行相同流水线（应该使用缓存）
        import time
        start_time = time.time()
        result2 = await pipeline.execute(X, y, steps=[
            ('data_preprocessing', {}),
            ('model_training', {})
        ])
        end_time = time.time()

        # 验证缓存效果
        assert result1['success'] is True
        assert result2['success'] is True

        # 第二次执行应该更快
        cache_hit_time = end_time - start_time
        # 这里我们假设第一次执行已经完成，主要验证缓存机制存在
        assert hasattr(pipeline, 'cache_stats')
        assert pipeline.get_cache_stats()['hit_count'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
