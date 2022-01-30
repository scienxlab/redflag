"""Test redflag"""
import pytest
import redflag as rf

def test_redflag():
    """Test the basics.
    """
    i = rf.furthest_distribution([3,0,0,1,2,3,2,3,2,3,1,1,2,3,3,4,3,4,3,4,])
    assert i ==  [0.8, 0, 0, 0.2, 0]
