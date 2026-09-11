# GitHub 更新与上传指南

本目录是待审阅的完整仓库候选版本。不要把 `gbm-demo-100.zip` 放进普通
Git 提交；它约 107 MiB，应作为 GitHub Release 附件上传。

## 上传前

1. 完成本目录中的验证，并检查 `VALIDATION_REPORT.md`。
2. 根据作者选择加入 `LICENSE`。
3. 在全新环境测试安装，补写安装耗时、CPU 和内存。
4. 冻结稿件图号，更新 `MANUSCRIPT_RESULTS_MAP.md`。

## 更新仓库代码

将本目录内容与 GitHub 仓库根目录对应。建议使用 Git 客户端或命令行提交
整个结构，不要在网页上逐个覆盖文件：

```text
README.md
MANUSCRIPT_RESULTS_MAP.md
NATURE_CODE_CHECKLIST.md
.gitignore
gbm-albumin-sieving/
sd-albumin-sieving/
gfb-integration/
```

### 推荐：用 GitHub Desktop

1. 在 GitHub Desktop 中克隆
   `zzz721/GFB-albumin-sieving-model`，不要直接修改网页上的 `main`。
2. 在客户端中新建分支，例如 `nature-code-update`。
3. 打开该仓库的本地目录。保留隐藏的 `.git` 目录，将本候选目录中的上述
   文件和三个软件目录复制到仓库根目录；不要把外层
   `GFB-albumin-sieving-model-update` 再套一层。
4. 如果候选目录中已经不再存在某个旧的受版本控制文件，在 GitHub Desktop
   的 **Changes** 页面确认它应删除后再提交。不要复制本地 `_review/`、
   `submission_materials/` 或任何 `outputs/`。
5. 提交说明可写：
   `Update Nature review code, demos, and conserved GFB integration`。
6. 推送该分支，在 GitHub 比较页面重点核对 README、双半径参数、SD 101 点
   设置、`gfb-integration/` 和删除文件列表，再合并到 `main`。

旧文件中与稿件算法不同但需要追溯的脚本已标记为 historical；候选目录仍
保留这些文件。不要在未检查调用关系时额外删除。

## 上传大型演示包

代码进入最终分支后，在 GitHub 仓库主页：

1. 打开 **Releases**。
2. 点击 **Draft a new release**。
3. 先确定标签名，例如 `v1.0.0-nature-review`，并把演示 README 中的
   占位符改成该标签对应的预期附件地址：
   `https://github.com/zzz721/GFB-albumin-sieving-model/releases/download/v1.0.0-nature-review/gbm-demo-100.zip`。
4. 提交这次 README 修改，然后让 Release 标签指向这个最终提交。
5. 填写 Release 标题与版本说明。
6. 将本地 `submission_materials/gbm-demo-100.zip` 拖到附件区域。
7. 整理期间选择 **Save draft**；核对附件名与预期地址一致后再
   **Publish release**。

最终提供给编辑和审稿人的入口建议使用 Release 页面链接；页面应同时展示
代码标签、说明和 `gbm-demo-100.zip` 附件。

GitHub 官方参考：

- [在 GitHub Desktop 中管理和发布分支](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop)
- [创建和管理 GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
