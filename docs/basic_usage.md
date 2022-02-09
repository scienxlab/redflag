# Basic usage

Redflag is a collection of functions. They currently run on single columns of data (one feature from `X` or one target of `y`). For example:

```python
import redflag as rf

data = [-3, -2, -2, -1, 0, 0, 0, 1, 2, 2, 3]

rf.has_outliers(data)
array([], dtype=int64)

rf.has_outliers(3 * data + [100])
array([100])
```
