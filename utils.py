import pandas as pd
import numpy as np
import re
from ast import literal_eval
import asyncio

# --- Utility Functions ---


def _rename_dynamic_date_columns(df: pd.DataFrame, suffix: str = None):
    # 获取所有包含日期的列名
    date_start_cols = [
        col for col in df.columns if re.match(r"\d{4}-\d{2}-\d{2}-(.*)", col)
    ]

    target_cols = [col for col in date_start_cols if col.rfind(suffix) != -1]

    # 提取日期并排序
    dates = sorted(
        [re.search(r"(\d{4}-\d{2}-\d{2})", col).group(1) for col in target_cols],
        reverse=True,
    )

    # 只处理最近两个交易日
    if len(dates) >= 1:
        df.columns = df.columns.str.replace(dates[0], "上一个交易日")
    if len(dates) >= 2:
        df.columns = df.columns.str.replace(dates[1], "上两个交易日")
        
def apply_filters_for_data_frame(
    df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:
    """
    通用Pandas筛选函数(字符串条件版)
    支持类似SQL WHERE子句的筛选条件语法

    示例:
        "pe > 20"
        "roe >= 0.15 AND turnoverRate < 5"
        "date >= '2023-01-01' AND close > open"
        "code in ['600000', '601318']"

    支持操作符:
        >, <, >=, <=, =, !=, in, not in, contains, not contains
    支持逻辑运算符:
        AND, OR, NOT
    支持括号分组:
        "(pe > 20 OR pe < 10) AND roe > 0.15"
    Args:
        df: 要筛选的DataFrame
        condition: 筛选条件字符串
    Returns:
        筛选后的DataFrame
    Raises:
        ValueError: 如果条件语法错误或列名不存在
    """
    if not condition:
        return df.copy()

    # 复制DataFrame避免修改原始数据
    df = df.copy()

    # 自动检测并转换数值列
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                # 尝试转换为数值类型
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except:
                pass

    # 解析条件字符串为Pandas查询表达式
    try:
        query_expr = _parse_condition_to_query(condition, df.columns)
        return df.query(query_expr)
    except Exception as e:
        raise ValueError(f"条件查询失败: {str(e)}") from e


def _parse_condition_to_query(condition: str, valid_columns) -> str:
    """
    将条件字符串解析为Pandas query表达式

    Args:
        condition: 原始条件字符串
        valid_columns: DataFrame的有效列名集合(可以是set或pandas Index)
    Returns:
        可用于DataFrame.query()的表达式字符串
    """
    import re
    from ast import literal_eval

    original_condition = condition

    # 确保valid_columns是set类型
    if not isinstance(valid_columns, set):
        valid_columns = set(valid_columns)

    # 预处理: 移除多余空格，统一操作符格式
    condition = re.sub(r"\s+", " ", condition.strip())

    # 定义逻辑运算符关键字
    LOGIC_KEYWORDS = {"AND", "OR", "NOT", "and", "or", "not"}

    # 改进的列名匹配模式，支持中文、特殊字符和百分号
    column_pattern = r"([a-zA-Z_\u4e00-\u9fa5][\w\u4e00-\u9fa5\-%]*)"
    columns = set(re.findall(column_pattern, condition)) - LOGIC_KEYWORDS

    # 检查列名是否有效
    invalid_cols = [col for col in columns if col not in valid_columns]
    if invalid_cols:
        raise ValueError(f"无效列名: {invalid_cols}")

    # 处理带引号的列名
    quoted_columns = re.findall(r'[\'"](.*?)[\'"]', condition)
    for col in quoted_columns:
        if col in valid_columns:
            condition = condition.replace(f"'{col}'", f"`{col}`").replace(
                f'"{col}"', f"`{col}`"
            )

    # 确保所有列名都被反引号包裹
    for col in columns:
        if col in valid_columns:
            # 处理包含特殊字符的列名
            if any(c in col for c in ["-", "%", " "]):
                # 获取原始列名(不带反引号)
                raw_col = col.strip("`")
                # 替换条件中的列名(带或不带反引号)
                condition = re.sub(
                    rf"(?<!\w)(`?{re.escape(raw_col)}`?)(?!\w)",
                    f"`{raw_col}`",
                    condition,
                )
            else:
                # 普通列名处理
                if (
                    f"`{col}`" not in condition
                    and f"'{col}'" not in condition
                    and f'"{col}"' not in condition
                ):
                    condition = re.sub(rf"(?<!\w){col}(?!\w)", f"`{col}`", condition)

    # 转换操作符 - 使用正则表达式确保精确匹配
    operator_patterns = [
        (r"(?<!\S)!=(?!\S)", "!="),  # != 保持不变
        (r"(?<!\S)=(?!\S)", "=="),  # = 替换为 ==
        (r"(?<!\S)>(?!\S)", ">"),  # > 保持不变
        (r"(?<!\S)<(?!\S)", "<"),  # < 保持不变
        (r"(?<!\S)>=(?!\S)", ">="),  # >= 保持不变
        (r"(?<!\S)<=(?!\S)", "<="),  # <= 保持不变
        (r"(?<!\S)in(?!\S)", ".isin("),  # in 替换为 .isin(
        (r"(?<!\S)not in(?!\S)", ".notin("),  # not in 替换为 .notin(
        (r"(?<!\S)contains(?!\S)", ".str.contains("),  # contains 替换
        (r"(?<!\S)not contains(?!\S)", ".str.contains("),  # not contains 替换
    ]

    # 处理字符串字面量
    condition = re.sub(r"'(.*?)'", r'"\1"', condition)  # 单引号转双引号

    # 处理IN表达式
    in_pattern = (
        r"(`?[a-zA-Z_\u4e00-\u9fa5][\w\u4e00-\u9fa5-%]*`?)\s+(in|not in)\s+(\[.*?\])"
    )

    def replace_in(match):
        col = match.group(1)
        # 处理包含特殊字符的列名
        if any(c in col for c in ["-", "%", " "]):
            # 确保列名被反引号包裹
            if not col.startswith("`") or not col.endswith("`"):
                col = f"`{col.strip('`')}`"
        else:
            # 普通列名处理
            if not col.startswith("`"):
                col = f"`{col}`"
        op = match.group(2)
        values = match.group(3)
        try:
            # 安全解析列表
            values = literal_eval(values)
            if not isinstance(values, list):
                raise ValueError
            values_str = str(values)
        except:
            raise ValueError(f"无效的列表格式: {values}")

        if op == "in":
            return f"`{col}`.isin({values_str})"
        else:
            return f"~`{col}`.isin({values_str})"

    condition = re.sub(in_pattern, replace_in, condition)

    # 处理逻辑运算符 - 统一转换为小写
    condition = re.sub(r"\bAND\b", "&", condition)
    condition = re.sub(r"\bOR\b", "|", condition)
    condition = re.sub(r"\bNOT\b", "~", condition)

    # 处理其他操作符
    for pattern, replacement in operator_patterns:
        condition = re.sub(pattern, replacement, condition)

    # 处理字符串包含操作
    condition = condition.replace(".str.contains(", ".str.contains(")
    condition = condition.replace(" not contains ", " not contains ")

    # 验证最终表达式语法
    try:
        # 使用空DataFrame测试语法(确保columns是list类型)
        test_df = pd.DataFrame(columns=list(valid_columns))
        test_df.query(condition)
    except Exception as e:
        # 提供更友好的错误提示
        error_msg = f"无效的条件表达式: {str(e)}\n"
        error_msg += f"原条件: {original_condition}\n"
        error_msg += f"转换后: {condition}\n"
        error_msg += "提示: 请检查列名、操作符和值格式是否正确"
        raise ValueError(error_msg) from e

    return condition


# 分位数计算, 1260对应5年的交易日
def __calculate_percentile(df, item_col_name, window=1260):
    df = df.copy()
    # 将date转换为DatetimeIndex
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # 数据清洗
    df[item_col_name] = df[item_col_name].replace([np.inf, -np.inf], np.nan)
    df[item_col_name] = df[item_col_name].interpolate(method="time").ffill().bfill()

    # 安全计算函数
    def safe_percentile(x):
        valid_data = x[~np.isnan(x)]
        if len(valid_data) < 60:
            return np.nan
        current = valid_data[-1]
        history = valid_data[:-1]
        return (current > history).mean() * 100

    # 滚动计算
    item_percentile = item_col_name + "_percentile"
    df[item_percentile] = (
        df[item_col_name]
        .rolling(window=window, min_periods=60)
        .apply(safe_percentile, raw=True)
    )
    # 处理缺失值
    df[item_percentile] = (
        df[item_percentile].interpolate(method="linear").ffill().bfill()
    )
    # 重置索引，恢复date列
    df = df.reset_index()
    return df


def __filter_abnormal_3delta(df, column_name):
    lower = df[column_name].mean() - 3 * df[column_name].std()
    upper = df[column_name].mean() + 3 * df[column_name].std()
    df = df[(df[column_name] > lower) & (df[column_name] < upper)]
    return df
