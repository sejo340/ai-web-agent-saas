import json
from playwright.async_api import async_playwright
from database import AsyncSessionLocal
from models import Website
from sqlalchemy import select


async def scrape_website_task(website_url: str, client_id: int):
    print(f"🤖 Starting background scrape for {website_url}")
    try:
        # 1. Scrape the website
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(website_url, timeout=15000)

            # Extract simple text from buttons and links
            elements = await page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a, button, input, h1, h2, p')).map(el => el.innerText).filter(t => t.length > 0).slice(0, 50);
            }""")
            await browser.close()

        # 2. Save the map to a file
        with open(f"client_{client_id}_map.json", "w") as f:
            json.dump(elements, f)
        print("✅ Map saved to JSON")

        # 3. Update the database safely (using its own connection)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Website).where(Website.client_id == client_id)
            )
            website = result.scalar_one_or_none()
            if website:
                website.bot_status = "ready"
                await db.commit()
                print(f"✅ Database updated to 'ready' for {website_url}")

    except Exception as e:
        print(f"❌ Scrape failed: {e}")
        # Try to mark as failed in DB
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Website).where(Website.client_id == client_id)
                )
                website = result.scalar_one_or_none()
                if website:
                    website.bot_status = "failed"
                    await db.commit()
        except:
            pass
