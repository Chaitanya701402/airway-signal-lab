# Bundled reconstruction validation report

**Status: passed.** This report validates the saved reconstruction arrays and manifest. It does not validate clinical meaning of the generated conditions.

## Checks performed

- Each source group occurs in exactly one split.
- Each split has matching feature, label and manifest counts.
- Features and labels are finite; labels are within the four declared classes.
- Manifest class counts match the encoded labels.
- No exact log-Mel feature hash appears across splits.
- The saved training features have mean 0 and standard deviation 1 after normalisation.

## Results

| Split | Windows | Source groups | normal | ambient | movable synthetic | fixed synthetic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 288 | 4 | 72 | 72 | 72 | 72 |
| val | 144 | 2 | 36 | 36 | 36 | 36 |
| test | 144 | 2 | 36 | 36 | 36 | 36 |

Exact feature hashes crossing splits: **0**.

## Scope

The samples are derived from eight bundled ICBHI breathing sources. Ambient and obstruction examples are controlled digital transformations, so this validation supports reproducibility and split hygiene within the reconstruction; it is not external or clinical validation.
