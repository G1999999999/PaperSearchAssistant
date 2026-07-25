from __future__ import annotations

import pytest

from eval.query_understanding_golden import assert_qu_case, load_qu_cases


@pytest.mark.parametrize("case", load_qu_cases(), ids=lambda c: c.id)
def test_golden_query_understanding(case):
    assert_qu_case(case)
