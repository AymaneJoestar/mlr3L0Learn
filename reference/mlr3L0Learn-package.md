# mlr3L0Learn: 'mlr3' Learner for L0-Regularized Regression via 'L0Learn'

Provides a regression learner for the 'mlr3' framework that wraps best
subset selection with L0, L0L1 and L0L2 penalties from the 'L0Learn'
package (Hazimeh and Mazumder, 2020,
[doi:10.1287/opre.2019.1919](https://doi.org/10.1287/opre.2019.1919) ).
The penalty parameters are selected by cross-validation inside
'L0Learn::L0Learn.cvfit()'.

## See also

Useful links:

- <https://github.com/AymaneJoestar/mlr3L0Learn>

- Report bugs at <https://github.com/AymaneJoestar/mlr3L0Learn/issues>

## Author

**Maintainer**: Aymane Aboulhouda <aymane.aboulhouda@usherbrooke.ca>

Authors:

- Aymane Aboulhouda <aymane.aboulhouda@usherbrooke.ca>
