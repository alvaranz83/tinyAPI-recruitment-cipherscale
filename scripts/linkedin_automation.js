// scripts/linkedin_automation.js
import puppeteer from "puppeteer";

// Credentials from environment
const username = process.env.LINKEDIN_EMAIL;
const password = process.env.LINKEDIN_PASSWORD;

/**
 * Utility to add timestamps to console logs
 */
function logWithTime(message, emoji = "🧩") {
  const now = new Date().toISOString().split("T")[1].replace("Z", "");
  console.log(`[${now}] ${emoji} ${message}`);
}

/**
 * Launch Puppeteer, login (click sign-in first if needed), and extract DOM for the given URL.
 */
export async function scrapePage(url) {
  logWithTime(`Visiting: ${url}`, "🌐");

  // 1️⃣ Launch browser
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();
  logWithTime("Puppeteer launched successfully", "🚀");

  // 2️⃣ Navigate to URL
  try {
    await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });
    logWithTime("Page loaded successfully", "✅");
  } catch (err) {
    logWithTime(`Error loading page: ${err.message}`, "❌");
  }

  
 // 3️⃣ Handle LinkedIn AUTHWALL — open the real "Sign In" form
  try {
    const currentUrl = page.url();
    logWithTime(`Checking current URL: ${currentUrl}`, "🔍");
  
    // ✅ Only proceed if we’re on the LinkedIn Authwall
    if (currentUrl.includes("linkedin.com/authwall")) {
      logWithTime("Detected LinkedIn AUTHWALL page — restricted content view.", "🚪");
  
      // Wait for the "Sign in" button to appear and become visible
      await page.waitForSelector(".authwall-join-form__form-toggle--bottom.form-toggle", {
        timeout: 15000,
        visible: true,
      });
  
      const authWallButton = await page.$(".authwall-join-form__form-toggle--bottom.form-toggle");
      if (!authWallButton) {
        logWithTime("⚠️ Could not find Authwall 'Sign in' button.", "⚠️");
      } else {
        logWithTime("🖱️ Found Authwall 'Sign in' button — preparing to click.", "🖱️");
  
        // Scroll into view before clicking
        await authWallButton.evaluate((el) =>
          el.scrollIntoView({ behavior: "smooth", block: "center" })
        );
  
        // Attempt click, with JS fallback if necessary
        try {
          await authWallButton.click({ delay: 100 });
          logWithTime("✅ Clicked Authwall 'Sign in' button successfully.", "✅");
        } catch (clickErr) {
          logWithTime(
            `⚠️ Click failed (${clickErr.message}) — retrying via JS click.`,
            "🛠️"
          );
          await page.evaluate(() => {
            const el = document.querySelector(".authwall-join-form__form-toggle--bottom.form-toggle");
            if (el) el.click();
          });
        }
  
        // Wait for either navigation or the real Authwall sign-in form to appear
        logWithTime("⏳ Waiting for Authwall Sign-in form or navigation...", "⌛");
        await Promise.race([
          page.waitForNavigation({ waitUntil: "networkidle2", timeout: 20000 }),
          page.waitForSelector('form[data-id="sign-in-form"], form.authwall-sign-in-form__body', {
            timeout: 20000,
            visible: true,
          }),
        ]);
  
        // Confirm login form presence
        const signInForm = await page.$('form[data-id="sign-in-form"], form.authwall-sign-in-form__body');
        if (signInForm) {
          logWithTime("🎉 Authwall Sign-in form detected — ready for credential input.", "🎉");
        } else {
          logWithTime("⚠️ Could not detect Authwall Sign-in form after click.", "⚠️");
        }
      }
    } else {
      logWithTime("ℹ️ Not on an AUTHWALL page — skipping this step.", "ℹ️");
    }
  } catch (err) {
    logWithTime(`❌ Error handling Authwall login: ${err.message}`, "❌");
  }
  


  // 4️⃣ Attempt login
  try {
    logWithTime("Waiting for email input...", "⌛");
    await page.waitForSelector(
      "input[type='email'], input[name='session_key'], input[name='username'], input[name='email']",
      { timeout: 15000 }
    );
  
    logWithTime("Typing email...", "📧");
    await page.type(
      "input[type='email'], input[name='session_key'], input[name='username'], input[name='email']",
      username,
      { delay: 50 }
    );
  
    logWithTime("Waiting for password input...", "🔑");
    await page.waitForSelector(
      "input[type='password'], input[name='session_password'], input[name='password']",
      { timeout: 15000 }
    );
  
    logWithTime("Typing password...", "🔒");
    await page.type(
      "input[type='password'], input[name='session_password'], input[name='password']",
      password,
      { delay: 50 }
    );
  
    // 🧠 Known submit button selectors
    const submitSelectors = [
      "button.sign-in-form__submit-btn--full-width",
      "button[data-id='sign-in-form__submit-btn']",
      "#join-form-submit",
      ".btn-primary.sign-in-form__submit-btn--full-width"
    ];
  
    let submitSelector = null;
    for (const sel of submitSelectors) {
      try {
        await page.waitForSelector(sel, { timeout: 5000, visible: true });
        submitSelector = sel;
        break;
      } catch {
        continue;
      }
    }
  
    if (submitSelector) {
      logWithTime(`Submitting login form using selector: ${submitSelector}`, "🚀");
      await page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, submitSelector);
  
      await page.waitForFunction(
        (sel) => {
          const el = document.querySelector(sel);
          if (!el) return false;
          const rect = el.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0;
        },
        { timeout: 8000 },
        submitSelector
      );
  
      try {
        await page.click(submitSelector, { delay: 100 });
        logWithTime("Clicked 'Sign in' submit button successfully", "✅");
      } catch (clickErr) {
        logWithTime(`⚠️ Click failed (${clickErr.message}), retrying via JS click`, "🛠️");
        await page.evaluate((sel) => {
          const el = document.querySelector(sel);
          if (el) el.click();
        }, submitSelector);
      }
    } else {
      logWithTime("⚠️ No visible submit button found — pressing Enter", "⚠️");
      await page.keyboard.press("Enter");
    }
  
    // ✅ Reliable login confirmation — LinkedIn top nav bar check
    logWithTime("Waiting for login confirmation (global-nav)...", "🔄");
  
    await page.waitForFunction(() => {
      const nav = document.querySelector(".global-nav__content");
      if (!nav) return false;
  
      const primaryItems = nav.querySelectorAll(".global-nav__primary-item");
      // Logged-in LinkedIn has between 6 and 8 main nav items
      return primaryItems.length >= 6 && primaryItems.length <= 8;
    }, { timeout: 30000 });
  
    logWithTime("✅ Login confirmed — LinkedIn global-nav detected (6–8 items).", "🎉");
  } catch (err) {
    logWithTime(`⚠️ Login skipped or failed: ${err.message}`, "⚠️");
  }


  // 5️⃣ Ensure full DOM loaded
  logWithTime("Waiting 3s for DOM to stabilize...", "⏳");
  await new Promise((r) => setTimeout(r, 3000));

  // 6️⃣ Extract structured DOM
  logWithTime("Extracting structured DOM...", "🧠");
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

  logWithTime("✅ DOM captured successfully", "📄");

  // Optional preview for logs (shortened)
  const snippet = JSON.stringify(domData).slice(0, 500);
  logWithTime(`DOM Preview: ${snippet}...`, "🔍");

  await browser.close();
  logWithTime("Browser closed successfully", "🧹");

  // Return serialized JSON for Python subprocess
  console.log("###DOM_JSON###" + JSON.stringify(domData));
}

// Allow running from CLI or subprocess
if (process.argv[2]) {
  const url = process.argv[2];
  scrapePage(url).catch((err) => {
    logWithTime(`Fatal error in scrapePage: ${err.message}`, "💥");
    console.error("❌ Error:", err);
    process.exit(1);
  });
}
