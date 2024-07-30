from bs4 import BeautifulSoup

# HTML字符串
html_data = '''<html><body><table><tbody><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr><tr><td>SegLink [26]</td><td>70.0</td><td>86.0</td><td>77.0</td><td>8.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PixelLink[4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake[18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField[37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>76.7</td><td>87.4</td><td>81.7</td><td></td></tr><tr><td>[3] FTSN</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><td></td></tr><tr><td>CRAFT[2]</td><td>78.2</td><td>88.2</td><td></td><td>8.6</td></tr><tr><td></td><td></td><td></td><td>82.9</td><td></td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DB[12] PAN[34]</td><td>79.2 83.8</td><td>91.5 84.4</td><td>84.9 84.1</td><td>32.0 30.2</td></tr><tr><td colspan="5">Ours(MLT-17) 12.31 85.57 86.62 84.54 (SynText) 12.68 82.97 85.40 80.68 Ours DRRG 85.08 88.05 82.30 [41]</td></tr></tbody></table></body></html>'''

'''
# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html_data, 'html.parser')

# 查找表格
table = soup.find('table')

# 初始化一个列表来存储所有行的数据
table_data = []

# 遍历表格中的所有行
for row in table.find_all('tr'):
    # 获取该行的所有单元格
    cols = row.find_all('td')
    # 获取每个单元格的文本，并去除空白字符
    cols = [ele.text.strip() for ele in cols]
    # 将该行的数据添加到列表中
    table_data.append(cols)

# 打印提取的数据
for data_row in table_data:
    print(data_row)
'''
soup = BeautifulSoup(html_data, 'html.parser')
table = soup.find('table')

# 遍历表格中的所有行
for row_index, row in enumerate(table.find_all('tr')):
    # 获取该行的所有单元格
    for col_index, cell in enumerate(row.find_all(['td', 'th'])):  # Include <th> if headers are also required
        # 获取单元格的文本
        cell_text = cell.get_text().strip()
        # 打印单元格文本和索引信息
        print(f"Cell at row {row_index}, col {col_index}: '{cell_text}' - Start Index: ({row_index}, {col_index}), End Index: ({row_index}, {col_index})")
