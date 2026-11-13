"""Render the two structural diagrams (architecture, matched-seed design)
from HTML to high-DPI PNG via Playwright, per charts-skill rules
(auto-fit viewport to bounding box, device_scale_factor=2)."""

import asyncio
import os
from playwright.async_api import async_playwright

BUILD = os.path.dirname(os.path.abspath(__file__))
JOBS = [
    (f"{BUILD}/diagram_architecture.html", f"{BUILD}/figures/fig_architecture.png", 1160),
    (f"{BUILD}/diagram_design.html", f"{BUILD}/figures/fig_design.png", 1080),
]


async def render(html_path, out_path, width, scale=2):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": width, "height": 900},
            device_scale_factor=scale,
        )
        await page.goto(f"file://{html_path}", wait_until="networkidle")
        await page.wait_for_timeout(400)
        el = page.locator("#root")
        bbox = await el.bounding_box()
        if bbox:
            fit_w = max(width, int(bbox["width"] + 40))
            fit_h = int(bbox["height"] + 40)
            await page.set_viewport_size({"width": fit_w, "height": fit_h})
            await page.wait_for_timeout(200)
        await el.screenshot(path=out_path)
        await browser.close()
        print(f"saved {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


async def main():
    for html, out, w in JOBS:
        await render(html, out, w)


if __name__ == "__main__":
    asyncio.run(main())
