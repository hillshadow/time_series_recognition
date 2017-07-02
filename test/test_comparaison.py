# -*- coding: utf-8 -*-

import comparison as comp

def test_MAPE():
    assert comp.MAPE([1,1,1,1], [1,1,1,1])==0
    