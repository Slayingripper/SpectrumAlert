# Publishing these pages to GitHub Wiki

GitHub hosts wiki pages in a separate repository. This directory is the
source-controlled copy of the wiki content.

1. Enable **Settings → Features → Wikis** on GitHub.
2. Clone the wiki repository:

   ```bash
   git clone https://github.com/Slayingripper/SpectrumAlert.wiki.git
   ```

3. Copy the Markdown files from this directory into the wiki clone, excluding
   this README.
4. Commit and push:

   ```bash
   cp docs/wiki/*.md /path/to/SpectrumAlert.wiki/
   cd /path/to/SpectrumAlert.wiki
   git add .
   git commit -m "Publish SpectrumAlert wiki"
   git push
   ```

`_Sidebar.md` and `_Footer.md` are recognised by GitHub Wiki and provide the
shared navigation and footer.
