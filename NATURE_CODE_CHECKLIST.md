# Nature code and software checklist status

Status date: 2026-09-11. This is an author-facing completion checklist.

| Checklist item | Current status | Required action before release |
|---|---|---|
| Source code | GBM, SD, and conserved-flow GFB modules published and validated from a fresh clone | Remove legacy tracked caches/outputs in the cleanup commit |
| Small demo data | GBM 100-network Release asset published; SD and GFB inputs included | Complete; keep the stable asset URL |
| Operating systems and dependency versions | Windows 11, Python 3.12.6 and tested package versions recorded | Record any additional environment only after testing it |
| Non-standard hardware | None required; timing host used an Intel Core i9-13980HX and 15.63 GiB RAM | Complete |
| Installation instructions | Present; fresh environment and package installation passed in approximately 2 minutes 18 seconds | Ask an unfamiliar colleague to repeat the instructions |
| Demo instructions and expected output | Present; public Release, GBM/SD/GFB outputs, and SD figures were checked | Complete |
| Use with new data | GBM raw-network classification and transport commands, workbook format, and SD width/geometry path documented and tested | Complete |
| Reproduction of manuscript results | Main result map created | Add final fitted inputs, figure source data, and remaining plotting workflows |
| Software license | Missing | Authors select an OSI-approved license and add `LICENSE` |
| Open repository link | Public repository and Release asset are accessible | Fix the remaining placeholder in the GBM example README |
| Detailed code description in manuscript | Methods contains GBM, SD, electrostatic, and serial GFB equations | Check final section names against this repository map |
| Colleague installation test | Not completed | Ask a colleague unfamiliar with the software to follow the final README |
| Single ZIP or reviewer-accessible link | Tagged Release and `gbm-demo-100.zip` are publicly accessible | Complete |

The post-release fresh-clone check completed the full SD geometry calculation
in approximately 63.8 seconds and the 100-network GBM demo in approximately
59.8 seconds with four workers. Fresh environment creation, exact dependency
installation, and installation of all three repository packages took
approximately 2 minutes 18 seconds in total on the recorded desktop.

The optional full quantitative reproduction instructions are not yet complete.
The repository should not be described as a complete one-command reproduction
of every manuscript figure until the remaining source-data and plotting entries
in `MANUSCRIPT_RESULTS_MAP.md` are resolved.
