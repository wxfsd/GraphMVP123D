import os
import time
import pandas as pd
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# === 文件路径配置 ===
input_smi_path = "/nfs_home/xiaofeng/zhen/GraphMVP_xf/src_regression/output/DTI/data/DrugBank_12316_uniqueFillter_12302.smi"
output_csv_path = "/nfs_home/xiaofeng/zhen/GraphMVP_xf/src_regression/output/DTI/DrugBank_12316_uniqueFillter_12302_for_sale.csv"
# debug_img_dir = "/nfs_home/xiaofeng/zhen/GraphMVP_xf/src_regression/output/DTI/debug_imgs"
# os.makedirs(debug_img_dir, exist_ok=True)

# === 初始化浏览器（Linux服务器适配） ===
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.binary_location = "/usr/bin/google-chrome"

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# === 核心判断函数 ===
def check_emolecules_supplier(smiles, idx):
    encoded = quote(smiles)
    url_hash = f"#?query={encoded}&system-type=BB&p=1"
    full_url = f"https://orderbb.emolecules.com/search/{url_hash}"

    try:
        # 访问主页，再设置 hash 并刷新，确保页面正确加载
        driver.get("https://orderbb.emolecules.com/search/")
        driver.execute_script(f"window.location.hash = '{url_hash}';")
        time.sleep(1.5)
        driver.refresh()

        # 等待“Please wait...”弹窗消失
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(text(),'Please wait')]"))
            )
            WebDriverWait(driver, 20).until(
                EC.invisibility_of_element_located((By.XPATH, "//div[contains(text(),'Please wait')]"))
            )
        except TimeoutException:
            pass

        # 等待数据表格加载
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
        )

        # # 调试截图保存
        # screenshot_path = os.path.join(debug_img_dir, f"debug_{idx+1}.png")
        # driver.save_screenshot(screenshot_path)

        # 提取供应商信息（第6列）
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        suppliers = set()
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 6:
                supplier_text = cols[5].text.strip()
                if supplier_text:
                    suppliers.add(supplier_text)

        if suppliers:
            return "available", len(suppliers), full_url
        else:
            return "not-available", 0, full_url

    except TimeoutException:
        print(f"[Timeout] {smiles}")
        return "timeout", 0, full_url
    except Exception as e:
        print(f"[Error] {smiles}: {e}")
        return "error", 0, full_url
 
# === 主体处理流程 ===
df = pd.read_csv(input_smi_path, header=None, names=["smiles"])
results = []

for i, smi in enumerate(df["smiles"]):
    status, count, url = check_emolecules_supplier(smi, i)
    results.append((smi, status, count, url))
    print(f"[{i+1}/{len(df)}] {smi} → {status} (Suppliers: {count})")
    time.sleep(1.0)  # 防止 IP 被封

# === 保存结果 ===
df_result = pd.DataFrame(results, columns=["smiles", "status", "supplier_count", "url"])
df_result.to_csv(output_csv_path, index=False)
print(f"\n✅ 处理完成，结果保存至：{output_csv_path}")

# === 关闭浏览器 ===
driver.quit()