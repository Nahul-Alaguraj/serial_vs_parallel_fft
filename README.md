﻿# serial_vs_parallel_fft
This report was done as a part of my Computer Architecture and Organization Assignment 
This report does a comparative study of computing Serial and parallel FFTs for small to medium range values using parallel python
## 📊 Key Highlights

- Serial FFT outperformed parallel in most cases due to multiprocessing overhead.
- Efficiency dropped below 1% for 4-core runs.
- NumPy’s internal optimization is sufficient for moderate signal sizes.
