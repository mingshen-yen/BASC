# BASC

## Overview
BASC (BASCO, BASeline COrrection) is a processing scheme extended from eBASCO (Schiappapietra et al., 2021) to remove the baseline of strong-motion records through piecewise linear detrending of the velocity time history. Unlike standard processing workflows, eBASCO does not apply frequency filtering to suppress low-frequency content. As a result, it preserves both (i) long-period, near-source ground motion—often expressed as a one-sided pulse in the velocity trace—and (ii) the final displacement offset (fling-step). The software is designed for the rapid identification of fling-containing waveforms in large strong-motion datasets.

## Features
- Efficient baseline correction tailored for time series data.
- User-friendly interface for easy integration.

## Usage
You can use the jupter-notebook directly for baseline correction.

## Input/Output
- **Input:** Time series data files in three component (E/N/Z) with `.mseed` formats.
- **Output:** Cleaned time series data ready for analysis with values in `.txt` formats.

## Method
BASC employs advanced algorithms to analyze the time series data and applies sophisticated techniques to correct for baseline drift while preserving the integrity of the data.

## Reproducibility
To ensure reproducibility, we provide clear documentation along with sample datasets that customers can use to validate the functionality of the BASC tool.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Reference
Schiappapietra, E., Felicetta, C., & D’Amico, M. (2021). Fling-Step Recovering from Near-Source Waveforms Database. Geosciences, 11(2), 67. https://doi.org/10.3390/geosciences11020067
