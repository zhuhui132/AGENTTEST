"""
负载测试
"""
import pytest
import time
import threading
import queue
import random
import statistics
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent import Agent

class TestLoadTesting:
    """负载测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.base_agent = Agent("负载测试助手")
        self.results = []
        self.errors = []

    def test_concurrent_users_load(self):
        """测试并发用户负载"""
        def user_simulation(user_id, message_count):
            """模拟用户行为"""
            agent = Agent(f"负载测试用户{user_id}")
            user_results = []

            try:
                for i in range(message_count):
                    message = f"用户{user_id}消息{i+1}"

                    # 随机添加一些变化
                    if random.random() < 0.3:
                        message = f"你好，{message}"

                    start_time = time.time()
                    result = agent.process_message(message)
                    end_time = time.time()

                    response_time = end_time - start_time
                    user_results.append({
                        "user_id": user_id,
                        "message_id": i,
                        "response_time": response_time,
                        "response_length": len(result["response"]),
                        "timestamp": end_time
                    })

                    # 模拟用户思考时间
                    time.sleep(random.uniform(0.1, 0.5))

            except Exception as e:
                self.errors.append(f"用户{user_id}错误: {str(e)}")

            return user_results

        # 启动并发用户
        user_count = 10
        messages_per_user = 5
        threads = []

        for user_id in range(user_count):
            thread = threading.Thread(
                target=user_simulation,
                args=(user_id, messages_per_user)
            )
            threads.append(thread)

        # 启动所有线程
        start_time = time.time()
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
        end_time = time.time()

        # 分析结果
        total_time = end_time - start_time
        total_requests = user_count * messages_per_user
        throughput = total_requests / total_time

        # 性能断言
        assert len(self.errors) == 0, f"负载测试出现错误: {self.errors}"
        assert throughput > 5.0  # 每秒至少处理5个请求
        assert total_time < 30.0  # 总时间不超过30秒

    def test_sustained_load(self):
        """测试持续负载"""
        def sustained_worker(worker_id, duration):
            """持续工作线程"""
            agent = Agent(f"持续负载工作器{worker_id}")
            start_time = time.time()
            request_count = 0

            try:
                while time.time() - start_time < duration:
                    message = f"持续负载消息{request_count}"

                    req_start = time.time()
                    result = agent.process_message(message)
                    req_end = time.time()

                    self.results.append({
                        "worker_id": worker_id,
                        "request_id": request_count,
                        "response_time": req_end - req_start,
                        "timestamp": req_end
                    })

                    request_count += 1

                    # 控制请求频率
                    time.sleep(0.2)

            except Exception as e:
                self.errors.append(f"工作器{worker_id}错误: {str(e)}")

        # 启动持续负载测试
        worker_count = 5
        test_duration = 10  # 10秒
        threads = []

        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=sustained_worker,
                args=(worker_id, test_duration)
            )
            threads.append(thread)

        # 启动测试
        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        end_time = time.time()

        # 分析结果
        actual_duration = end_time - start_time
        total_requests = len(self.results)

        if total_requests > 0:
            avg_response_time = statistics.mean(r["response_time"] for r in self.results)
            p95_response_time = statistics.quantiles(
                [r["response_time"] for r in self.results], n=20
            )[18]  # 95th percentile
            throughput = total_requests / actual_duration
        else:
            avg_response_time = 0
            p95_response_time = 0
            throughput = 0

        # 性能断言
        assert len(self.errors) == 0, f"持续负载测试出现错误: {self.errors}"
        assert total_requests > 0, "没有处理任何请求"
        assert avg_response_time < 2.0, f"平均响应时间过长: {avg_response_time}s"
        assert p95_response_time < 5.0, f"95%响应时间过长: {p95_response_time}s"
        assert throughput > 2.0, f"吞吐量过低: {throughput} req/s"

    def test_peak_load(self):
        """测试峰值负载"""
        def peak_load_burst(burst_id, request_count):
            """峰值负载突发"""
            agent = Agent(f"峰值负载代理{burst_id}")
            burst_results = []

            try:
                for i in range(request_count):
                    message = f"峰值负载消息{burst_id}_{i}"

                    start_time = time.time()
                    result = agent.process_message(message)
                    end_time = time.time()

                    burst_results.append({
                        "burst_id": burst_id,
                        "request_id": i,
                        "response_time": end_time - start_time,
                        "timestamp": end_time
                    })

            except Exception as e:
                self.errors.append(f"峰值负载{burst_id}错误: {str(e)}")

            return burst_results

        # 模拟峰值负载
        burst_count = 20
        requests_per_burst = 10
        threads = []

        start_time = time.time()

        # 同时启动多个突发负载
        for burst_id in range(burst_count):
            thread = threading.Thread(
                target=peak_load_burst,
                args=(burst_id, requests_per_burst)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # 分析峰值负载结果
        total_time = end_time - start_time
        total_requests = burst_count * requests_per_burst
        peak_throughput = total_requests / total_time

        # 性能断言
        assert len(self.errors) == 0, f"峰值负载测试出现错误: {self.errors}"
        assert peak_throughput > 20.0, f"峰值吞吐量过低: {peak_throughput} req/s"
        assert total_time < 60.0, f"峰值负载处理时间过长: {total_time}s"

    def test_memory_under_load(self):
        """测试负载下的内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        def memory_load_test(worker_id, iterations):
            """内存负载测试"""
            try:
                for i in range(iterations):
                    agent = Agent(f"内存测试代理{worker_id}_{i}")

                    # 发送一些消息增加内存使用
                    for j in range(5):
                        long_message = f"这是来自工作器{worker_id}的第{i}轮第{j}条测试消息，" * 10
                        result = agent.process_message(long_message)

                        # 检查响应
                        assert "response" in result

                    # 手动触发垃圾回收（如果可能）
                    del agent

            except Exception as e:
                self.errors.append(f"内存测试工作器{worker_id}错误: {str(e)}")

        # 启动内存负载测试
        worker_count = 5
        iterations_per_worker = 20
        threads = []

        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=memory_load_test,
                args=(worker_id, iterations_per_worker)
            )
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        end_time = time.time()

        # 检查内存使用
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        test_duration = end_time - start_time

        # 内存断言
        assert len(self.errors) == 0, f"内存负载测试出现错误: {self.errors}"
        assert memory_increase < 200 * 1024 * 1024, f"内存增长过多: {memory_increase / 1024 / 1024:.2f}MB"
        assert test_duration < 60.0, f"内存测试时间过长: {test_duration}s"

class TestStressTesting:
    """压力测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.test_results = []
        self.test_errors = []

    def test_resource_exhaustion(self):
        """测试资源耗尽情况"""
        def resource_exhaustion_worker(worker_id, max_requests):
            """资源耗尽测试工作器"""
            try:
                agent = Agent(f"资源耗尽测试{worker_id}")

                for i in range(max_requests):
                    # 发送越来越复杂的消息
                    message = f"复杂测试消息{worker_id}_{i} " * (i // 10 + 1)

                    start_time = time.time()
                    result = agent.process_message(message)
                    end_time = time.time()

                    self.test_results.append({
                        "worker_id": worker_id,
                        "request_id": i,
                        "response_time": end_time - start_time,
                        "message_length": len(message),
                        "timestamp": end_time
                    })

                    # 逐渐减少延迟，增加压力
                    time.sleep(max(0.01, 0.1 - i * 0.01))

            except Exception as e:
                self.test_errors.append(f"工作器{worker_id}在请求{i}时出错: {str(e)}")

        # 启动资源耗尽测试
        worker_count = 8
        requests_per_worker = 50
        threads = []

        start_time = time.time()

        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=resource_exhaustion_worker,
                args=(worker_id, requests_per_worker)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # 分析压力测试结果
        total_time = end_time - start_time
        successful_requests = len(self.test_results)
        failed_requests = len(self.test_errors)
        total_requests = worker_count * requests_per_worker

        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        if successful_requests > 0:
            avg_response_time = statistics.mean(r["response_time"] for r in self.test_results)
            max_response_time = max(r["response_time"] for r in self.test_results)
        else:
            avg_response_time = 0
            max_response_time = 0

        # 压力测试断言
        assert success_rate > 0.8, f"成功率过低: {success_rate:.2%}"
        assert avg_response_time < 5.0, f"平均响应时间过长: {avg_response_time}s"
        assert max_response_time < 30.0, f"最大响应时间过长: {max_response_time}s"
        assert failed_requests < total_requests * 0.2, f"失败请求过多: {failed_requests}"

    def test_extreme_concurrency(self):
        """测试极限并发"""
        def extreme_concurrency_test(thread_id):
            """极限并发测试"""
            try:
                agent = Agent(f"极限并发代理{thread_id}")

                # 每个线程只发送少量请求，但并发数很高
                for i in range(3):
                    message = f"极限并发测试{thread_id}_{i}"

                    start_time = time.time()
                    result = agent.process_message(message)
                    end_time = time.time()

                    self.test_results.append({
                        "thread_id": thread_id,
                        "request_id": i,
                        "response_time": end_time - start_time,
                        "timestamp": end_time
                    })

            except Exception as e:
                self.test_errors.append(f"极限并发线程{thread_id}错误: {str(e)}")

        # 启动极限并发测试
        thread_count = 50  # 50个并发线程
        threads = []

        start_time = time.time()

        for thread_id in range(thread_count):
            thread = threading.Thread(
                target=extreme_concurrency_test,
                args=(thread_id,)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # 分析极限并发结果
        total_time = end_time - start_time
        total_requests = thread_count * 3
        successful_requests = len(self.test_results)

        if successful_requests > 0:
            avg_response_time = statistics.mean(r["response_time"] for r in self.test_results)
            throughput = successful_requests / total_time
        else:
            avg_response_time = 0
            throughput = 0

        # 极限并发断言
        assert len(self.test_errors) == 0, f"极限并发出现错误: {self.test_errors}"
        assert successful_requests > 0, "没有成功处理的请求"
        assert avg_response_time < 10.0, f"极限并发响应时间过长: {avg_response_time}s"
        assert throughput > 5.0, f"极限并发吞吐量过低: {throughput} req/s"

class TestPerformanceMetrics:
    """性能指标测试"""

    def test_response_time_distribution(self):
        """测试响应时间分布"""
        response_times = []
        agent = Agent("响应时间测试")

        # 收集响应时间数据
        for i in range(100):
            message = f"响应时间测试消息{i}"

            start_time = time.time()
            result = agent.process_message(message)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            # 随机延迟模拟真实使用
            time.sleep(random.uniform(0.05, 0.2))

        # 计算统计指标
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        min_time = min(response_times)
        max_time = max(response_times)

        # 响应时间分布断言
        assert avg_time < 2.0, f"平均响应时间过长: {avg_time:.3f}s"
        assert median_time < 1.5, f"中位数响应时间过长: {median_time:.3f}s"
        assert p95_time < 5.0, f"95%响应时间过长: {p95_time:.3f}s"
        assert p99_time < 10.0, f"99%响应时间过长: {p99_time:.3f}s"
        assert min_time > 0, "最小响应时间应该大于0"
        assert max_time < 20.0, f"最大响应时间过长: {max_time:.3f}s"

    def test_throughput_measurement(self):
        """测试吞吐量测量"""
        def throughput_worker(worker_id, request_count, results_queue):
            """吞吐量工作器"""
            agent = Agent(f"吞吐量测试{worker_id}")
            worker_times = []

            for i in range(request_count):
                message = f"吞吐量测试{worker_id}_{i}"

                start_time = time.time()
                result = agent.process_message(message)
                end_time = time.time()

                worker_times.append(end_time - start_time)

                # 控制请求频率
                time.sleep(0.1)

            results_queue.put(worker_times)

        # 启动吞吐量测试
        worker_count = 3
        requests_per_worker = 20
        results_queue = queue.Queue()
        threads = []

        start_time = time.time()

        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=throughput_worker,
                args=(worker_id, requests_per_worker, results_queue)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()

        # 收集结果
        all_response_times = []
        while not results_queue.empty():
            worker_times = results_queue.get()
            all_response_times.extend(worker_times)

        # 计算吞吐量
        total_time = end_time - start_time
        total_requests = worker_count * requests_per_worker
        throughput = total_requests / total_time

        # 吞吐量断言
        assert len(all_response_times) == total_requests, "响应时间数据不完整"
        assert throughput > 5.0, f"吞吐量过低: {throughput:.2f} req/s"
        assert total_time < 60.0, f"吞吐量测试时间过长: {total_time:.2f}s"
