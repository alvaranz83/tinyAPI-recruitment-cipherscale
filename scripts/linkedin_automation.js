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

  // 3️⃣ Look for "Sign in" / "Log in" button
  try {
    const signInSelector = await page.evaluate(() => {
      const lower = (t) => t?.toLowerCase() || "";
      const links = Array.from(document.querySelectorAll("a, button"));
  
      // Priority known LinkedIn cases
      const modalButton = document.querySelector(".sign-in-modal");
      if (modalButton) return ".sign-in-modal";
  
      const headerButton = document.querySelector(".nav__button-secondary.btn-primary.btn-md");
      if (headerButton) return ".nav__button-secondary.btn-primary.btn-md";
  
      // Fallback generic detection
      for (const el of links) {
        const text = lower(el.textContent);
        const href = lower(el.getAttribute("href"));
        const id = lower(el.id);
        const cls = lower(el.className);
  
        if (
          text.includes("sign in") ||
          text.includes("log in") ||
          href?.includes("login") ||
          href?.includes("signin") ||
          id?.includes("login") ||
          id?.includes("signin") ||
          cls?.includes("sign-in")
        ) {
          // Choose safest selector available
          if (el.id) return `#${el.id}`;
          if (el.className) {
            const safeClass = "." + el.className.trim().split(/\s+/).join(".");
            return `${el.tagName.toLowerCase()}${safeClass}`;
          }
          if (el.getAttribute("href")) return `a[href='${el.getAttribute("href")}']`;
        }
      }
      return null;
    });
  
    if (signInSelector) {
      logWithTime(`Found 'Sign in' element: ${signInSelector}`, "🖱️");
      await page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, signInSelector);
      await page.click(signInSelector, { delay: 100 });
      await new Promise((r) => setTimeout(r, 4000));
      logWithTime("Clicked 'Sign in' and waited 4s", "⏱️");
    } else {
      logWithTime("No 'Sign in' button found — maybe already on login page", "⚠️");
    }
  } catch (err) {
    logWithTime(`Error finding/clicking 'Sign in': ${err.message}`, "⚠️");
  }

  // 4️⃣ Attempt login
  try {
    logWithTime("Waiting for email input...", "⌛");
    await page.waitForSelector(
      "input[type='email'], input[name='session_key'], input[name='username'], input[name='email']",
      { timeout: 12000 }
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
      { timeout: 12000 }
    );
  
    logWithTime("Typing password...", "🔒");
    await page.type(
      "input[type='password'], input[name='session_password'], input[name='password']",
      password,
      { delay: 50 }
    );
  
    // 🧠 Known submit buttons (LinkedIn changes this often)
    const submitSelectors = [
      "button.sign-in-form__submit-btn--full-width",
      "button[data-id='sign-in-form__submit-btn']",
      "#join-form-submit", // legacy
      ".btn-primary.sign-in-form__submit-btn--full-width"
    ];
  
    let submitSelector = null;
  
    // Wait until *any* of these appear
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
  
      // Ensure the element is interactable
      await page.evaluate((sel) => {
        const el = document.querySelector(sel);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, submitSelector);
  
      // Double-check that the element is attached & visible
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
  
      // Try safe click first, fallback to JS click
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
  
    logWithTime("Waiting for post-login navigation...", "🔄");
    await page.waitForNavigation({ waitUntil: "networkidle2", timeout: 30000 });
    logWithTime("✅ Login successful or redirected.", "✅");
  } catch (err) {
    logWithTime(`Login skipped or failed: ${err.message}`, "⚠️");
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
