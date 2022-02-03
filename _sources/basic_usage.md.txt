# Basic usage

```python
import redflag as rf

data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]

rf.has_outliers(data)
array([], dtype=int64)

rf.has_outliers(3 * data + [100])
array([100])
```
