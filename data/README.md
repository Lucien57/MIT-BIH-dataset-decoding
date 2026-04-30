Place the two NumPy files here before training:
available at https://cloud.tsinghua.edu.cn/d/78ae667b037c4fad97c0/

- `dataset_raw.npy`: shape `(92192, 300)` or `(92192, 1, 300)`
- `labelset_raw.npy`: shape `(92192,)`, integer labels in `{0, 1, 2, 3, 4}`

The default config assumes five MIT-BIH heartbeat classes:

| label | class |
|---:|---|
| 0 | N: Normal beat |
| 1 | A: Atrial premature beat |
| 2 | V: Ventricular premature beat |
| 3 | L: Left bundle branch block |
| 4 | R: Right bundle branch block |
