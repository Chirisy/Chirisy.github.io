# 部署说明

这个站点是纯静态 docsify 项目，不需要构建步骤。

## 本地预览

```bash
node serve.js
```

访问：

```text
http://localhost:3000
```

## GitHub Pages

1. 推送代码到 GitHub 仓库。
2. 打开仓库 `Settings -> Pages`。
3. Source 选择 `Deploy from a branch`。
4. Branch 选择 `main`，目录选择 `/root`。
5. 保存后等待 Pages 部署完成。

仓库根目录的 `.nojekyll` 用于避免 GitHub Pages 忽略以下划线开头的 docsify 文件，例如 `_sidebar.md` 和 `_navbar.md`。
