// scripts/scraper.js
import puppeteer from "puppeteer";

const username = process.env.LINKEDIN_EMAIL;
const password = process.env.LINKEDIN_PASSWORD;

/**
 * Launch Puppeteer, login (click sign-in first if needed), and extract DOM for the given URL.
 */
export async function scrapePage(url) {
  console.log(`🌐 Visiting: ${url}`);

  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();

  await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });

  // 1️⃣ Try to find and click "Sign in"
  try {
    const signInButton = await page.$x(
      "//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'sign in') or " +
        "contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'log in') or " +
        "contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'login') or " +
        "contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'signin') or " +
        "contains(@id,'login') or contains(@id,'signin')]"
    );
    if (signInButton.length > 0) {
      console.log("🖱️ Clicking 'Sign in' button...");
      await signInButton[0].click();
      await page.waitForTimeout(4000);
    } else {
      console.log("⚠️ No visible 'Sign in' button found.");
    }
  } catch (err) {
    console.log("⚠️ Error clicking 'Sign in' button:", err.message);
  }

  // 2️⃣ Fill in credentials
  try {
    await page.waitForSelector(
      "input[type='email'], input[name='session_key'], input[name='username'], input[name='email']",
      { timeout: 8000 }
    );
    await page.type(
      "input[type='email'], input[name='session_key'], input[name='username'], input[name='email']",
      username,
      { delay: 50 }
    );

    await page.waitForSelector(
      "input[type='password'], input[name='session_password'], input[name='password']",
      { timeout: 8000 }
    );
    await page.type(
      "input[type='password'], input[name='session_password'], input[name='password']",
      password,
      { delay: 50 }
    );

    const submitButton = await page.$x(
      "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'sign in') or " +
        "contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'log in') or " +
        "contains(@type,'submit') or contains(@id,'login')]"
    );
    if (submitButton.length > 0) {
      console.log("🚀 Submitting login form...");
      await submitButton[0].click();
    } else {
      console.log("⚠️ No submit button found — pressing Enter instead.");
      await page.keyboard.press("Enter");
    }

    await page.waitForNavigation({ waitUntil: "networkidle2", timeout: 30000 });
    console.log("✅ Logged in successfully.");
  } catch (err) {
    console.log("⚠️ No login form detected or failed to fill credentials:", err.message);
  }

  // 3️⃣ Wait for DOM to fully load
  await page.waitForTimeout(3000);

  // 4️⃣ Extract structured DOM
  const domData = await page.evaluate(() => {
    function serializeNode(node) {
      const obj = {
        tag: node.tagName,
        attributes: {},
        text: node.childElementCount === 0 ? node.textContent.trim() : null,
        children: [],
      };
      if (node.attributes) {
        for (const attr of node.attributes) {
          obj.attributes[attr.name] = attr.value;
        }
      }
      for (const child of node.children) {
        obj.children.push(serializeNode(child));
      }
      return obj;
    }
    return serializeNode(document.body);
  });

  console.log("✅ DOM captured successfully.");
  console.log(JSON.stringify(domData, null, 2));

  await browser.close();

  // Return serialized JSON as a single line (for Python parsing)
  console.log("###DOM_JSON###" + JSON.stringify(domData));
}

// Allow running from CLI or subprocess
if (process.argv[2]) {
  const url = process.argv[2];
  scrapePage(url).catch((err) => {
    console.error("❌ Error:", err);
    process.exit(1);
  });
}
