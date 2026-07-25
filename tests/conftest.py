def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: 需要向量库、LLM 或 RUN_LAYER_INTEGRATION=1",
    )
