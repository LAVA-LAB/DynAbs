Changelog
==============

This changelog lists changes that could impact the behavior and usage of the code.

Aug 25, 2024
-------------
- Bug fixes in plotting functions and monte carlo simulations.
- Repaired bugs in the block refinement scheme.
- Cleaned and archived code for PhD thesis.

Aug 19, 2024
-------------
- Upgraded to PRISM 4.8.1, which uses a slightly different output format for policies (time=action instead of time:action).

Feb 21, 2024
-------------
- The `path_to_prism.txt` file is now removed. Instead, the user should provide an additional `prism_executable` argument when running the code, which points to the prism executable. See the ReadMe for further details.