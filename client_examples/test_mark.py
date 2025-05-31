import requests

url = "http://localhost:8765/mark"
payload = {
    "seamless": """1：对于所有方法，我们使用了10交叉验证（即每一折中有556个训练样本和62个测试样本）来调整自由参数，例如 GPOR 和 LapSVM 的核函数形式和参数。请注意，所有的替代方法都将 X 和 Z 堆叠在一起形成一个整体的数据矩阵，而忽略了它们的异质性。<br>
2：与某个垂直领域一一对应的特征（如 Clarity、ReDDE、给定垂直领域查询日志的查询可能性以及 Soft.ReDDE）在缩放之前已在所有垂直领域之间归一化。监督训练/测试是通过10交叉验证完成的。参数 τ 是在与我们单一特征基线相同的500个查询验证集上为每一个训练折调优的。<br>""",
    "prompt": """1：对于所有方法，我们使用了10折交叉验证（即每一折中有556个训练样本和62个测试样本）来调整自由参数，例如 GPOR 和 LapSVM 的核函数形式和参数。请注意，所有的替代方法都将 X 和 Z 堆叠在一起形成一个整体的数据矩阵，而忽略了它们的异质性。<br>
2：与某个垂直领域一一对应的特征（如 Clarity、ReDDE、给定垂直领域查询日志的查询可能性以及 Soft.ReDDE）在缩放之前已在所有垂直领域之间归一化。监督训练/测试是通过10折交叉验证完成的。参数 τ 是在与我们单一特征基线相同的500个查询验证集上为每一个训练折调优的。<br>
""",
    "lang": "zh"
}
headers = {
    "Content-Type": "application/json; charsetUTF-8"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raises HTTPError if the request returned an unsuccessful status code
    data = response.json()
    print("Marked Text:", data["marked_translations"])
except requests.exceptions.RequestException as e:
    print("Error:", e)
    
    
