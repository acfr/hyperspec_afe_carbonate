# Carbonate Feature Extraction from Hyperspectral Image Cube

This Python script extracts and visualises carbonate features from a hyperspectral image cube using Automated Feature Extraction (AFE), as described by Murphy et al. (2017).

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

To run the script on the example data located in the data/ directory:

```bash
python ImagePlotAFE.py
```

The example data located in the data/ directory was acquired from a coral sample provided by H. V. McGregor.  See McGregor and Gagan (2003) for more information.

## Output

The script will generate visualisations of extracted carbonate features, which will be saved as an image file and displayed interactively.

## Reference

> McGregor, H. V., and M. K. Gagan (2003), Diagenesis and geochemistry of porites corals from Papua New Guinea: Implications for paleoclimate reconstruction, Geochim. Cosmochim. Acta, 67(12), 2147–2156.

> Murphy, R. J, Webster,  J.M., Nothdurft, L., Dechnik, B, McGregor, H. V., Patterson, M. A., Sanborn, K, Webb, G. E., Kearney, L. I., Rintoul, L., Erler, D. V., 2017. High resolution hyperspectral imaging of diagenesis and clays in fossil coral reef material: a non-destructive tool for improving environmental and climate reconstructions, Geochem. Geophys. Geosyst.18, doi:10.1002/2017GC006949.
