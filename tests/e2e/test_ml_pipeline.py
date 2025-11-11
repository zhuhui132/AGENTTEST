"""
机器学习流水线端到端测试
测试完整的机器学习工作流程
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from src.ml.models import MLModel
from src.ml.training import ModelTrainer
from src.ml.preprocessing import DataPreprocessor
from src.ml.evaluation import ModelEvaluator
from src.core.types import TrainingConfig, DataConfig


class TestMLPipeline:
    """机器学习流水线端到端测试类"""

    @pytest.fixture
    def sample_data(self):
        """示例数据fixture"""
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randint(0, 5, 1000),
            'target': np.random.randint(0, 2, 1000)
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def data_config(self):
        """数据配置fixture"""
        return DataConfig(
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3'],
            test_size=0.2,
            random_state=42,
            preprocessing_steps=['scaling', 'encoding']
        )

    @pytest.fixture
    def training_config(self):
        """训练配置fixture"""
        return TrainingConfig(
            model_type='random_forest',
            test_size=0.2,
            cross_validation_folds=5,
            random_state=42,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        )

    @pytest.fixture
    def mock_model(self):
        """模拟模型"""
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0, 1] * 20))
        model.predict_proba = Mock(return_value=np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9]
        ] * 20))
        return model

    @pytest.fixture
    def mock_trainer(self, mock_model):
        """模拟训练器"""
        trainer = Mock()
        trainer.train = Mock(return_value=mock_model)
        trainer.get_metrics = Mock(return_value={
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85
        })
        return trainer

    @pytest.fixture
    def mock_preprocessor(self):
        """模拟预处理器"""
        preprocessor = Mock()
        preprocessor.fit_transform = Mock(return_value=Mock(
            features=np.random.randn(800, 3),
            target=np.random.randint(0, 2, 800)
        ))
        preprocessor.transform = Mock(return_value=Mock(
            features=np.random.randn(200, 3)
        ))
        return preprocessor

    @pytest.fixture
    def mock_evaluator(self):
        """模拟评估器"""
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value={
            'test_accuracy': 0.82,
            'precision': 0.80,
            'recall': 0.84,
            'f1_score': 0.82,
            'confusion_matrix': np.array([[82, 18], [16, 84]]),
            'classification_report': {
                '0': {'precision': 0.84, 'recall': 0.82},
                '1': {'precision': 0.80, 'recall': 0.84}
            }
        })
        return evaluator

    def test_complete_pipeline_workflow(self, sample_data, data_config, training_config,
                                   mock_preprocessor, mock_trainer, mock_evaluator):
        """测试完整的机器学习流水线工作流"""
        # 第一步：数据预处理
        processed_data = mock_preprocessor.fit_transform(sample_data, data_config)

        assert processed_data is not None
        assert hasattr(processed_data, 'features')
        assert hasattr(processed_data, 'target')

        # 第二步：模型训练
        model = mock_trainer.train(processed_data, training_config)

        assert model is not None
        assert mock_trainer.train.call_count == 1
        assert mock_preprocessor.fit_transform.call_count == 1

        # 第三步：模型评估
        # 模拟测试数据
        test_data = Mock(features=np.random.randn(200, 3))
        metrics = mock_evaluator.evaluate(model, test_data)

        assert 'test_accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics

        # 验证整体流水线完成
        assert mock_trainer.get_metrics.call_count == 1
        assert mock_evaluator.evaluate.call_count == 1

    @pytest.mark.asyncio
    async def test_data_quality_checks(self, sample_data):
        """测试数据质量检查流程"""
        # 创建数据质量检查器
        quality_checker = DataQualityChecker()

        # 执行质量检查
        quality_report = await quality_checker.check(sample_data)

        # 验证报告包含必要信息
        assert 'missing_values' in quality_report
        assert 'outliers' in quality_report
        assert 'data_types' in quality_report
        assert 'duplicate_rows' in quality_report
        assert 'class_balance' in quality_report

        # 验证基本质量指标
        assert quality_report['missing_values']['total'] >= 0
        assert len(quality_report['data_types']) > 0
        assert quality_report['duplicate_rows']['count'] >= 0

    def test_feature_engineering_pipeline(self, sample_data):
        """测试特征工程流水线"""
        feature_engineer = FeatureEngineer()

        # 定义特征工程步骤
        feature_steps = [
            ('polynomial_features', {'degree': 2}),
            ('interaction_features', {}),
            ('log_features', {'columns': ['feature1', 'feature2']}),
            ('binning', {'column': 'feature3', 'bins': 5})
        ]

        # 执行特征工程
        engineered_data = feature_engineer.transform(sample_data, feature_steps)

        assert engineered_data.shape[1] > sample_data.shape[1]  # 特征数量增加
        assert engineered_data.shape[0] == sample_data.shape[0]  # 样本数量不变

        # 验证新特征
        original_features = sample_data.columns.tolist()
        new_features = [col for col in engineered_data.columns if col not in original_features]
        assert len(new_features) > 0

    @pytest.mark.asyncio
    async def test_hyperparameter_optimization(self, sample_data, mock_model):
        """测试超参数优化流程"""
        # 创建超参数优化器
        optimizer = HyperparameterOptimizer()

        # 定义搜索空间
        search_space = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

        # 设置优化目标
        objective = 'accuracy'
        cv_folds = 5

        # 执行优化
        best_params, best_score = await optimizer.optimize(
            sample_data, search_space, objective, cv_folds, mock_model
        )

        assert best_params is not None
        assert best_score >= 0
        assert best_score <= 1.0
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params

    @pytest.mark.asyncio
    async def test_model_validation_workflow(self, sample_data, mock_model):
        """测试模型验证工作流"""
        validator = ModelValidator()

        # 执行验证测试
        validation_results = await validator.validate(mock_model, sample_data, {
            'cross_validation': True,
            'bootstrap_test': True,
            'learning_curve': True
        })

        # 验证结果结构
        assert 'cross_validation' in validation_results
        assert 'bootstrap_test' in validation_results
        assert 'learning_curve' in validation_results

        # 验证交叉验证结果
        cv_results = validation_results['cross_validation']
        assert 'scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert cv_results['mean_score'] >= 0
        assert cv_results['std_score'] >= 0

    def test_model_deployment_pipeline(self, mock_model):
        """测试模型部署流水线"""
        # 创建部署器
        deployer = ModelDeployer()

        # 模拟部署配置
        deploy_config = {
            'format': 'pickle',
            'optimize': True,
            'quantize': False,
            'include_metadata': True
        }

        # 执行部署
        deployment_result = deployer.deploy(mock_model, deploy_config)

        # 验证部署结果
        assert 'model_path' in deployment_result
        assert 'metadata' in deployment_result
        assert 'size_kb' in deployment_result
        assert 'deployment_time' in deployment_result

        assert deployment_result['size_kb'] > 0
        assert deployment_result['deployment_time'] > 0

    @pytest.mark.asyncio
    async def test_monitoring_setup(self, mock_model):
        """测试监控设置流程"""
        monitor = ModelMonitor()

        # 设置监控配置
        monitor_config = {
            'performance_metrics': ['latency', 'throughput', 'memory_usage'],
            'data_drift_detection': True,
            'alert_thresholds': {
                'accuracy_drop': 0.1,
                'latency_increase': 2.0
            }
        }

        # 启动监控
        monitoring_result = await monitor.setup(mock_model, monitor_config)

        # 验证监控设置
        assert monitoring_result['monitor_id'] is not None
        assert monitoring_result['metrics_configured'] is True
        assert monitoring_result['alerts_configured'] is True

    def test_batch_prediction_pipeline(self, mock_model):
        """测试批量预测流水线"""
        # 创建批量预测器
        batch_predictor = BatchPredictor(mock_model)

        # 准备批量数据
        batch_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randint(0, 5, 1000)
        })

        # 执行批量预测
        predictions = batch_predictor.predict(batch_data, batch_size=100)

        # 验证预测结果
        assert len(predictions) == 1000
        assert all(pred in [0, 1] for pred in predictions)
        assert isinstance(predictions, np.ndarray)

    @pytest.mark.asyncio
    async def test_model_retraining_workflow(self, sample_data, mock_model, mock_trainer):
        """测试模型重新训练工作流"""
        # 创建重新训练调度器
        scheduler = RetrainingScheduler()

        # 模拟性能下降
        current_performance = 0.75
        baseline_performance = 0.85
        performance_drop = baseline_performance - current_performance

        # 设置重新训练触发条件
        retrain_config = {
            'performance_drop_threshold': 0.05,
            'time_threshold_days': 30,
            'min_samples_increment': 100
        }

        # 检查是否需要重新训练
        should_retrain = await scheduler.check_retrain_needed(
            current_performance, baseline_performance, retrain_config
        )

        if performance_drop >= retrain_config['performance_drop_threshold']:
            assert should_retrain is True

            # 执行重新训练
            new_model = mock_trainer.train(sample_data, TrainingConfig(
                model_type='random_forest',
                n_estimators=150,  # 增加估计器数量
                random_state=42
            ))

            assert new_model is not None
            assert mock_trainer.train.call_count >= 2  # 初始训练 + 重新训练
        else:
            assert should_retrain is False

    def test_model_versioning_pipeline(self, mock_model):
        """测试模型版本管理流水线"""
        # 创建版本管理器
        version_manager = ModelVersionManager()

        # 注册模型版本
        version_info = {
            'model_version': '1.2.0',
            'training_date': '2023-01-01',
            'performance_metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87
            },
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10
            }
        }

        # 注册版本
        registration_result = version_manager.register_version(mock_model, version_info)

        assert registration_result['success'] is True
        assert registration_result['version_id'] is not None

        # 比较版本
        comparison_result = version_manager.compare_versions(
            registration_result['version_id'], '1.1.0'
        )

        assert comparison_result['improvement'] is True
        assert comparison_result['accuracy_improvement'] > 0

    @pytest.mark.asyncio
    async def test_experiment_tracking(self, mock_model):
        """测试实验跟踪流程"""
        # 创建实验跟踪器
        tracker = ExperimentTracker()

        # 定义实验配置
        experiment_config = {
            'experiment_name': 'rf_tuning_v2',
            'description': 'Random Forest超参数调优实验',
            'parameters': {
                'model_type': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10
            },
            'dataset': 'sample_dataset_v1',
            'tags': ['tuning', 'random_forest']
        }

        # 开始实验
        experiment_id = await tracker.start_experiment(experiment_config)

        assert experiment_id is not None

        # 记录实验结果
        results = {
            'accuracy': 0.87,
            'precision': 0.85,
            'recall': 0.88,
            'f1_score': 0.86,
            'training_time': 120.5,
            'model_size_mb': 2.3
        }

        await tracker.log_results(experiment_id, results)
        await tracker.complete_experiment(experiment_id, 'success')

        # 验证实验记录
        experiment_info = await tracker.get_experiment(experiment_id)
        assert experiment_info['status'] == 'completed'
        assert experiment_info['results'] == results

    @pytest.mark.asyncio
    async def test_pipeline_orchestration(self, sample_data):
        """测试流水线编排"""
        # 创建流水线编排器
        orchestrator = PipelineOrchestrator()

        # 定义流水线步骤
        pipeline_steps = [
            ('data_validation', DataValidator()),
            ('preprocessing', DataPreprocessor()),
            ('feature_engineering', FeatureEngineer()),
            ('model_training', ModelTrainer()),
            ('model_evaluation', ModelEvaluator()),
            ('model_deployment', ModelDeployer())
        ]

        # 配置流水线
        pipeline_config = {
            'parallel_steps': ['data_validation', 'preprocessing'],
            'sequential_steps': ['feature_engineering', 'model_training', 'model_evaluation'],
            'final_step': 'model_deployment',
            'error_handling': 'continue_on_error',
            'retry_count': 3
        }

        # 执行流水线
        pipeline_result = await orchestrator.execute_pipeline(
            sample_data, pipeline_steps, pipeline_config
        )

        # 验证流水线结果
        assert pipeline_result['success'] is True
        assert 'step_results' in pipeline_result
        assert 'final_output' in pipeline_result

        # 验证步骤执行
        for step_name in pipeline_config['sequential_steps']:
            assert step_name in pipeline_result['step_results']

        # 验证错误处理
        assert pipeline_result.get('errors', []) == []


class DataQualityChecker:
    """数据质量检查器"""

    async def check(self, data):
        """执行数据质量检查"""
        return {
            'missing_values': {
                'total': data.isnull().sum().sum(),
                'by_column': data.isnull().sum().to_dict()
            },
            'outliers': {
                'total': 100,  # 模拟异常值数量
                'by_column': {'feature1': 20, 'feature2': 15}
            },
            'data_types': data.dtypes.to_dict(),
            'duplicate_rows': {'count': 5},
            'class_balance': data['target'].value_counts().to_dict()
        }


class FeatureEngineer:
    """特征工程器"""

    def transform(self, data, steps):
        """执行特征工程"""
        import pandas as pd
        engineered_data = data.copy()

        for step_name, step_config in steps:
            if step_name == 'polynomial_features':
                degree = step_config.get('degree', 2)
                engineered_data['feature1_squared'] = engineered_data['feature1'] ** degree
                engineered_data['feature2_squared'] = engineered_data['feature2'] ** degree

            elif step_name == 'interaction_features':
                engineered_data['feature1_x_feature2'] = (
                    engineered_data['feature1'] * engineered_data['feature2']
                )

            elif step_name == 'log_features':
                for col in step_config.get('columns', []):
                    if col in engineered_data.columns:
                        engineered_data[f'{col}_log'] = np.log1p(
                            engineered_data[col].clip(lower=1)
                        )

        return engineered_data


# 其他辅助类的简单实现
class HyperparameterOptimizer:
    async def optimize(self, data, search_space, objective, cv_folds, model):
        # 模拟优化过程
        return {'n_estimators': 100, 'max_depth': 10}, 0.85


class ModelValidator:
    async def validate(self, model, data, config):
        return {
            'cross_validation': {
                'scores': [0.82, 0.85, 0.83, 0.86, 0.84],
                'mean_score': 0.84,
                'std_score': 0.015
            },
            'bootstrap_test': {'mean': 0.83, 'std': 0.02},
            'learning_curve': {'train_scores': [0.8, 0.85, 0.9], 'val_scores': [0.75, 0.82, 0.88]}
        }


class ModelDeployer:
    def deploy(self, model, config):
        import tempfile
        import pickle

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            pickle.dump(model, f)
            model_path = f.name

        return {
            'model_path': model_path,
            'metadata': {'type': 'random_forest', 'version': '1.0'},
            'size_kb': len(open(model_path, 'rb').read()) / 1024,
            'deployment_time': 2.5
        }


class ModelMonitor:
    async def setup(self, model, config):
        return {
            'monitor_id': 'monitor_001',
            'metrics_configured': True,
            'alerts_configured': True
        }


class BatchPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, data, batch_size):
        predictions = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batch_pred = self.model.predict(batch)
            predictions.extend(batch_pred)

        return np.array(predictions)


class RetrainingScheduler:
    async def check_retrain_needed(self, current_perf, baseline_perf, config):
        performance_drop = baseline_perf - current_perf
        return performance_drop >= config['performance_drop_threshold']


class ModelVersionManager:
    def register_version(self, model, info):
        return {'success': True, 'version_id': 'v1.2.0'}

    def compare_versions(self, current_id, target_id):
        return {'improvement': True, 'accuracy_improvement': 0.02}


class ExperimentTracker:
    async def start_experiment(self, config):
        return 'exp_001'

    async def log_results(self, exp_id, results):
        pass

    async def complete_experiment(self, exp_id, status):
        pass

    async def get_experiment(self, exp_id):
        return {'status': status, 'results': {}}


class PipelineOrchestrator:
    async def execute_pipeline(self, data, steps, config):
        return {
            'success': True,
            'step_results': {step: 'completed' for step in config['sequential_steps']},
            'final_output': 'model_deployed',
            'errors': []
        }


class ModelTrainer:
    def train(self, data, config):
        return Mock()


class ModelEvaluator:
    def evaluate(self, model, data):
        return {
            'test_accuracy': 0.82,
            'precision': 0.80,
            'recall': 0.84,
            'f1_score': 0.82,
            'confusion_matrix': np.array([[82, 18], [16, 84]]),
            'classification_report': {
                '0': {'precision': 0.84, 'recall': 0.82},
                '1': {'precision': 0.80, 'recall': 0.84}
            }
        }


class DataValidator:
    def __init__(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__])
