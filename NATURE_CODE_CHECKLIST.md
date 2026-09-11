# Nature code and software checklist status

Status date: 2026-09-11. This is an author-facing completion checklist.

| Checklist item | Current status | Required action before release |
|---|---|---|
| Source code | GBM, SD, and conserved-flow GFB modules assembled; local validation passed | Review and publish the staged commit |
| Small demo data | GBM 100-network archive complete; SD and GFB inputs included | Attach `gbm-demo-100.zip` to the final GitHub Release |
| Operating systems and dependency versions | Windows 11, Python 3.12.6 and tested package versions recorded | Record any additional environment only after testing it |
| Non-standard hardware | None required | Record CPU and RAM used for reported timing |
| Installation instructions | Present in package READMEs | Test from a clean environment and record installation time |
| Demo instructions and expected output | Present for GBM, SD, and GFB | Final cross-check after choosing the release tag |
| Use with new data | GBM workbook format and SD width/geometry path documented | Add or document the upstream GBM classification-generation path |
| Reproduction of manuscript results | Main result map created | Add final fitted inputs, figure source data, and remaining plotting workflows |
| Software license | Missing | Authors select an OSI-approved license and add `LICENSE` |
| Open repository link | Existing public repository | Replace demo URL placeholder after publishing the Release |
| Detailed code description in manuscript | Methods contains GBM, SD, electrostatic, and serial GFB equations | Check final section names against this repository map |
| Colleague installation test | Not completed | Ask a colleague unfamiliar with the software to follow the final README |
| Single ZIP or reviewer-accessible link | GBM demo ZIP ready; repository update staged | Publish a final tagged Release and provide its stable URL |

The full SD geometry calculation was timed at approximately 70.1 seconds in
the recorded Windows/Python environment. The 100-network GBM demo completed
200 calculations in approximately 62.0 seconds with four workers. These are
runtime measurements in the existing environment, not clean-install times.

The optional full quantitative reproduction instructions are not yet complete.
The repository should not be described as a complete one-command reproduction
of every manuscript figure until the remaining source-data and plotting entries
in `MANUSCRIPT_RESULTS_MAP.md` are resolved.
