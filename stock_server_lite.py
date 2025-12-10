import asyncio
import datetime
import operator
import re
import time
from typing import Dict, List, Union
from fastmcp import FastMCP
import pandas as pd
import akshare as ak
import numpy as np
from pydantic import Field
from utils import (
    apply_filters_for_data_frame,
    _parse_condition_to_query,
    __calculate_percentile,
    __filter_abnormal_3delta,
    _rename_dynamic_date_columns,
)

mcp = FastMCP("stock MCP", dependencies=["pandas", "numpy", "akshare"])

START_DATE = "20200101"
END_DATE = "20250418"


@mcp.tool()
async def get_hk_stock_info(
    stock_code: str,
    adjust: str = "qfq",
    condition: str = None,
):
    """
    根据股票代码获取港股的历史数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_code": {
          "type": "string",
          "description": "港股代码，如：'00700'",
          "examples": ["00700", "09988"]
        },
        "adjust": {
          "type": "string",
          "description": "复权方式：qfq-前复权, hfq-后复权, \"\"-不复权",
          "enum": ["qfq", "hfq", ""],
          "default": "qfq"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > 300 AND volume > 1000000"
          ]
        }
      },
      "required": ["stock_code"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量

    Examples:
        ```python
        await get_hk_stock_info(
            stock_code="00700",
            adjust="qfq",
            condition="close > 300"
        )
        ```
    """
    try:
        df = ak.stock_hk_daily(symbol=stock_code, adjust=adjust)
        if condition:
            return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
        return df.head(10).to_markdown()
    except Exception as e:
        print(f"港股 {stock_code} 数据获取失败：{str(e)}")
        return None


@mcp.tool()
async def get_us_stock_info(
    stock_code: str,
    adjust: str = "qfq",
    condition: str = None,
):
    """
    根据股票代码获取美股的历史数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_code": {
          "type": "string",
          "description": "美股代码，如：'AAPL'",
          "examples": ["AAPL", "TSLA"]
        },
        "adjust": {
          "type": "string",
          "description": "复权方式：qfq-前复权, hfq-后复权, \"\"-不复权",
          "enum": ["qfq", "hfq", ""],
          "default": "qfq"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > 150 AND volume > 1000000"
          ]
        }
      },
      "required": ["stock_code"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量

    Examples:
        ```python
        await get_us_stock_info(
            stock_code="AAPL",
            adjust="qfq",
            condition="close > 150"
        )
        ```
    """
    try:
        df = ak.stock_us_daily(symbol=stock_code, adjust=adjust)
        if condition:
            return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
        return df.head(10).to_markdown()
    except Exception as e:
        print(f"美股 {stock_code} 数据获取失败：{str(e)}")
        return None


@mcp.tool()
async def get_hk_index_info(
    symbol: str = "HSI",
    condition: str = None,
):
    """
    获取港股指数历史数据 (默认恒生指数)

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "指数代码，如：'HSI' (恒生指数), 'HSCEI' (恒生国企指数), 'HSCCI' (恒生红筹指数)",
          "default": "HSI"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串",
          "examples": [
            "close > 20000"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量

    Examples:
        ```python
        await get_hk_index_info(
            symbol="HSI",
            condition="close > 20000"
        )
        ```
    """
    try:
        df = ak.stock_hk_index_daily_sina(symbol=symbol)
        if condition:
            return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
        return df.head(10).to_markdown()
    except Exception as e:
        print(f"港股指数 {symbol} 获取失败：{str(e)}")
        return None


@mcp.tool()
async def get_us_index_info(
    symbol: str = ".IXIC",
    condition: str = None,
):
    """
    获取美股指数历史数据 (默认纳斯达克)

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "指数代码，如：'.IXIC' (纳斯达克), '.DJI' (道琼斯), '.INX' (标普500)",
          "default": ".IXIC"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串",
          "examples": [
            "close > 10000"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量

    Examples:
        ```python
        await get_us_index_info(
            symbol=".IXIC",
            condition="close > 10000"
        )
        ```
    """
    try:
        df = ak.index_us_stock_sina(symbol=symbol)
        if condition:
            return apply_filters_for_data_frame(df, condition).head(10).to_markdown()
        return df.head(10).to_markdown()
    except Exception as e:
        print(f"美股指数 {symbol} 获取失败：{str(e)}")
        return None


@mcp.tool()
async def get_single_stock_info(
    stock_code: str,
    start_date: str,
    end_date: str,
    retry: int = 3,
    condition: str = None,
):
    """
    根据股票代码获取股票的历史数据，包含PE、ROE等指标

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_code": {
          "type": "string",
          "description": "股票代码，如：'600000'",
          "examples": ["600000", "000001"]
        },
        "start_date": {
          "type": "string",
          "description": "开始日期，格式yyyyMMdd",
          "pattern": "^\\d{8}$",
          "examples": ["20200101", "20230101"]
        },
        "end_date": {
          "type": "string",
          "description": "结束日期，格式yyyyMMdd",
          "pattern": "^\\d{8}$",
          "examples": ["20201231", "20231231"]
        },
        "retry": {
          "type": "integer",
          "description": "请求失败后的重试次数",
          "default": 3,
          "minimum": 1,
          "maximum": 5
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "pe > 20 AND roe >= 0.15",
            "close > open AND volume > 100000"
          ]
        }
      },
      "required": ["stock_code", "start_date", "end_date"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量
        - code: 股票代码
        - transAmount: 成交额
        - amplitude: 振幅
        - riseFall: 涨跌幅(%)
        - riseFallAmount: 涨跌额
        - turnoverRate: 换手率(%)
        - pe: 滚动市盈率(TTM)
        - roe: 净资产收益率(%)

    Examples:
        ```python
        await get_single_stock_info(
            stock_code="600000",
            start_date="20230101",
            end_date="20231231",
            condition="pe < 15 AND roe > 0.1"
        )
        ```

    Error Handling:
        - 如果股票代码无效或数据获取失败，返回None
        - 如果条件语法错误，抛出ValueError
    """
    try:
        # 获取价格数据 (带重试)
        for _ in range(retry):
            try:
                # === 获取价格数据 ===
                price_df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    adjust="hfq",
                    start_date=start_date,
                    end_date=end_date,
                ).rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "股票代码": "code",
                        "成交额": "transAmount",
                        "振幅": "amplitude",
                        "涨跌幅": "riseFall",
                        "涨跌额": "riseFallAmount",
                        "换手率": "turnoverRate",
                    }
                )

                # 新浪财经接口（获取市盈率）
                pe_df = ak.stock_index_pe_lg()

                # 财报数据接口（获取ROE）
                finance_df = ak.stock_financial_abstract(symbol=stock_code)
                finance_df = finance_df[
                    finance_df["指标"].str.contains("ROE")
                    & finance_df["选项"].str.contains("盈利能力")
                ].copy()
                finance_df = finance_df.drop(columns=["选项"])
                report_date_cols = [
                    col for col in finance_df.columns if col.isdigit and len(col) == 8
                ]
                finance_df = pd.melt(
                    finance_df,
                    var_name="报表日期",
                    value_name="净资产收益率(ROE)",
                    value_vars=report_date_cols,
                )
                finance_df["date"] = pd.to_datetime(finance_df["报表日期"])
                finance_df["roe"] = finance_df["净资产收益率(ROE)"] / 100

                # === 数据合并 ===
                # 合并PE数据
                price_df["date"] = pd.to_datetime(price_df["date"])
                pe_df["日期"] = pd.to_datetime(pe_df["日期"])
                """
                市盈率指标：
                ╒══════════════════╤══════════════════════════╤═══════════════╤════════════╤═════════════════════════╕
                │ 指标名称         │ 公式                    │ 时间维度      │ 加权方式   │ 适用场景                │
                ╞══════════════════╪══════════════════════════╪═══════════════╪════════════╪═════════════════════════╡
                │ 静态市盈率       │ 总市值 / 上年净利润      │ 历史年度数据  │ 市值加权   │ 长期稳定盈利的公司      │
                ├──────────────────┼──────────────────────────┼───────────────┼────────────┼─────────────────────────┤
                │ 滚动市盈率(TTM)  │ 总市值 / 近4季度净利润   │ 最近12个月    │ 市值加权   │ 盈利波动大的行业        │
                ├──────────────────┼──────────────────────────┼───────────────┼────────────┼─────────────────────────┤
                │ 等权静态市盈率   │ 各股市值简单平均计算      │ 历史年度数据  │ 简单平均   │ 观察中小盘股估值        │
                ├──────────────────┼──────────────────────────┼───────────────┼────────────┼─────────────────────────┤
                │ 等权滚动市盈率   │ 各股市值简单平均计算      │ 最近12个月    │ 简单平均   │ 分析市场整体估值泡沫    │
                ╘══════════════════╧══════════════════════════╧═══════════════╧════════════╧═════════════════════════╛
                """
                merged_df = pd.merge_asof(
                    price_df.sort_values("date"),
                    pe_df[["日期", "等权滚动市盈率"]].rename(
                        columns={"日期": "date", "等权滚动市盈率": "pe"}
                    ),
                    on="date",
                    direction="backward",
                )

                # 合并ROE数据
                merged_df = pd.merge_asof(
                    merged_df.sort_values("date"),
                    finance_df[["date", "roe"]].sort_values("date"),
                    on="date",
                    direction="backward",
                )

                # === 数据清洗 ===
                merged_df["pe"] = merged_df["pe"].replace([np.inf, -np.inf], np.nan)
                merged_df["pe"] = merged_df["pe"].fillna(merged_df["pe"].median())
                merged_df["roe"] = merged_df["roe"].ffill()
                # 应用筛选条件
                if condition:
                    filtered_df = apply_filters_for_data_frame(merged_df, condition)
                    return filtered_df.to_markdown()
                return merged_df.head(10).to_markdown()
            except Exception as e:
                print(f"\n{stock_code}价格数据获取失败:{e}，重试中...")
                time.sleep(1)
    except Exception as e:
        print(f"{stock_code} 数据获取失败：{str(e)}\r\n")
        return None


@mcp.tool()
async def get_stock_indicator(
    stock_code: str, indicator_type: str = "沪深300", condition: str = None
):
    """
    根据股票代码获取指数历史行情数据，包含PE、PB和对应的百分位等

    Input Schema:
    {
      "type": "object",
      "properties": {
        "stock_code": {
          "type": "string",
          "description": "股票代码，如：'600000'",
          "examples": ["600000", "000001"]
        },
        "indicator_type": {
          "type": "string",
          "description": "指数类型",
          "enum": ["上证50", "沪深300", "上证380", "创业板50", "中证500", "上证180", "深证红利", "深证100", "中证1000", "上证红利", "中证100", "中证800"],
          "default": "沪深300"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "pe > 20 AND pb >= 0.5",
            "pe_percentile > 20 OR pb_percentile > 20"
          ]
        }
      },
      "required": ["stock_code"]
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - pe: 滚动市盈率(TTM)
        - pe_percentile: 市盈率历史百分位(近5年)
        - pb: 市净率
        - pb_percentile: 市净率历史百分位(近5年)

    Examples:
        ```python
        await get_stock_indicator(
            stock_code="600000",
            indicator_type="沪深300",
            condition="pe > 20 AND pb >= 0.5"
        )
        ```

    Error Handling:
        - 如果股票代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    # 获取基础行情
    df_price = _load_index_daily_with_fallback(stock_code)
    df_price["date"] = pd.to_datetime(df_price["date"]) 

    # 估值数据获取
    # 市盈率数据
    df_pe = _get_stock_pe(indicator_type)
    df_pe["date"] = pd.to_datetime(df_pe["date"])

    # 市净率数据
    df_pb = _get_stock_pb(indicator_type)

    df_pb["date"] = pd.to_datetime(df_pb["date"])

    # 融合PE
    merged_df = pd.merge(
        df_price, df_pe[["date", "pe_percentile"]], on="date", how="left"
    )
    merged_df["pe_percentile"] = merged_df["pe_percentile"].ffill().bfill()

    # 融合PB
    merged_df = pd.merge(
        merged_df, df_pb[["date", "pb_percentile"]], on="date", how="left"
    )
    merged_df["pb_percentile"] = merged_df["pb_percentile"].ffill().bfill()
    if condition:
        filtered_df = apply_filters_for_data_frame(merged_df, condition=condition)
        return filtered_df.to_markdown()
    return merged_df.to_markdown()




def _get_stock_pe(stock_type: str = "沪深300"):
    """
    根据股票类型获取对应市场的历史PE数据(等权滚动市盈率)
    Args:
        stock_type: PE、PB历史指标类型，根据股票所处的分类区分，可选值有：上证50, 沪深300, 上证380, 创业板50, 中证500, 上证180, 深证红利, 深证100, 中证1000, 上证红利, 中证100, 中证800
    Returns: PE数据, 返回字段有：date、pe、pe_percentile
    """
    # 新版指数估值接口
    # 市盈率PE = 股票价格 / 每股（年度）盈利；
    # 较高的PE意味着市场认为该股票有更好的盈利增长潜力
    df = ak.stock_index_pe_lg(stock_type)

    # 数据清洗
    df = df[["日期", "等权滚动市盈率"]].rename(
        columns={"日期": "date", "等权滚动市盈率": "pe"}
    )
    df["date"] = pd.to_datetime(df["date"])

    # 前向填充法
    df["pe"] = df["pe"].ffill()

    # 异常值过滤，3σ 法则
    df = __filter_abnormal_3delta(df, "pe")
    df = __calculate_percentile(df, "pe", 1260)
    return df


def _get_stock_pb(stock_type: str = "沪深300"):
    """
    根据股票类型获取对应市场的历史PB数据(等权市净率)

    Args:
        stock_type: 股票类型，可选值：
            "上证50", "沪深300", "上证380", "创业板50",
            "中证500", "上证180", "深证红利", "深证100",
            "中证1000", "上证红利", "中证100", "中证800"
    Returns:
        DataFrame包含字段：
        - date: 日期
        - pb: 等权市净率
        - pb_percentile: 历史百分位（近5年）

    使用示例：
    ```python
    _get_stock_pb("沪深300")
    ```
    """
    # 市净率PB = 股票价格 / 每股净资产
    # 用于衡量投资者为获得每一元净资产愿意支付多少元股价，较低的PB意味着市场低估了该公司的净资产
    pb_df = ak.stock_index_pb_lg(stock_type)

    # 数据清洗
    pb_df = pb_df[["日期", "等权市净率"]].rename(
        columns={"日期": "date", "等权市净率": "pb"}
    )
    pb_df["date"] = pd.to_datetime(pb_df["date"])

    # 前向填充法
    pb_df["pb"] = pb_df["pb"].ffill()

    # 异常值过滤，3σ 法则
    pb_df = __filter_abnormal_3delta(pb_df, "pb")
    pb_df = __calculate_percentile(pb_df, "pb", 1260)
    return pb_df



def _load_index_daily_with_fallback(code: str) -> pd.DataFrame:
    candidates: List[str] = []
    if code.startswith(("sh", "sz")):
        candidates = [code]
    else:
        candidates = [code, f"sh{code}", f"sz{code}"]
    last_err = None
    for c in candidates:
        try:
            df = ak.stock_zh_index_daily(c)
            if df is None or len(df) == 0:
                continue
            if "date" not in df.columns:
                df = df.rename(
                    columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
                )
            if "date" in df.columns:
                return df
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]) 



@mcp.tool()
async def fund_name_em():
    """
    获取东方财富网站-天天基金网-基金数据-所有基金的名称和类型
    数据来源：https://fund.eastmoney.com/manager/default.html#dt14;mcreturnjson;ftall;pn20;pi1;scabbname;stasc
    Returns:
      所有基金的名称和类型
    """
    df = ak.fund_name_em()
    return df.to_markdown()


@mcp.tool()
async def fund_info_index_em(
    symbol: str = "沪深指数", indicator: str = "被动指数型", condition: str = None
):
    """
    获取指数型基金近几年的单位净值、增长率、手续费等

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "行业类型",
          "enum": ["全部", "沪深指数", "行业主题", "大盘指数", "中盘指数", "小盘指数", "股票指数", "债券指数"],
          "default": "沪深指数"
        },
        "indicator": {
          "type": "string",
          "description": "基金类型",
          "enum": ["全部", "被动指数型", "增强指数型"],
          "default": "被动指数型"
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "近3年 > 5 AND 近2年 > 5",
            "日期 >= '2023-01-01'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 基金代码: 基金唯一标识
        - 基金名称: 基金全称
        - 单位净值: 最新单位净值
        - 日期: 净值日期
        - 日增长率: 单日增长率(%)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 手续费: 申购费率(%)
        - 起购金额: 最低申购金额(元)

    Examples:
        ```python
        await fund_info_index_em(
            symbol="沪深指数",
            indicator="被动指数型",
            condition="近3年 > 5 AND 手续费 < 1.5"
        )
        ```

    Error Handling:
        - 如果参数类型无效，抛出ValueError
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_info_index_em(symbol, indicator)
    sort_col = None
    for c in ["近1周", "近1月", "近3月", "近1年"]:
        if c in df.columns:
            sort_col = c
            break
    if sort_col:
        df = df.sort_values(by=sort_col, ascending=False)
    if condition:
        return apply_filters_for_data_frame(df, condition=condition).head(10).to_markdown()
    return df.head(10).to_markdown()



@mcp.tool()
async def fund_etf_spot_ths(date: str = "", condition: str = None) -> pd.DataFrame:
    """
    ETF-实时行情（同花顺）

    Input Schema:
    {
      "type": "object",
      "properties": {
        "date": {
          "type": "string",
          "description": "日期(yyyyMMdd)，为空返回当前最新数据",
          "default": ""
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "增长率 > 1 AND 基金类型 = '股票型'",
            "最新-交易日 >= '2025-01-01'"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号
        - 基金代码
        - 基金名称
        - 当前-单位净值
        - 当前-累计净值
        - 前一日-单位净值
        - 前一日-累计净值
        - 增长值
        - 增长率(%)
        - 赎回状态
        - 申购状态
        - 最新-交易日
        - 最新-单位净值
        - 最新-累计净值
        - 基金类型
        - 查询日期

    Examples:
        ```python
        await fund_etf_spot_ths(date="20240620")
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_spot_ths(date=date)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()


@mcp.tool()
async def fund_etf_hist_em(
    symbol: str = "159707",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
    condition: str = None,
) -> pd.DataFrame:
    """
    获取特定基金的每日ETF行情数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "ETF代码",
          "examples": ["159707", "510300"]
        },
        "period": {
          "type": "string",
          "description": "时间周期",
          "enum": ["daily", "weekly", "monthly"],
          "default": "daily"
        },
        "start_date": {
          "type": "string",
          "description": "开始日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20200101", "20230101"]
        },
        "end_date": {
          "type": "string",
          "description": "结束日期(yyyyMMdd)",
          "pattern": "^\\d{8}$",
          "examples": ["20201231", "20231231"]
        },
        "adjust": {
          "type": "string",
          "description": "复权方式",
          "enum": ["", "qfq", "hfq"],
          "default": ""
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > open AND volume > 100000",
            "date >= '2023-01-01'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: ETF 代码
        period: 类型，选项有：{'daily', 'weekly', 'monthly'}
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式，选项有：{"qfq": "前复权", "hfq": "后复权", "": "不复权"}
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "收盘 > open AND 成交量 > 100000"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 日期
        - 开盘
        - 收盘
        - 最高
        - 最低
        - 成交量
        - 成交额
        - 振幅
        - 涨跌幅
        - 涨跌额
        - 换手率

    Examples:
        ```python
        await fund_etf_hist_em(
            symbol="159707",
            condition="收盘 > open AND 成交量 > 100000"
        )
        ```

    Error Handling:
        - 如果ETF代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_hist_em(symbol, period, start_date, end_date, adjust)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()



@mcp.tool()
async def fund_etf_hist_sina(
    symbol: str = "sh510050", condition: str = None
) -> pd.DataFrame:
    """
    基金的日行情数据

    Input Schema:
    {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "基金代码",
          "examples": ["sh510050", "sz159915"]
        },
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法",
          "examples": [
            "close > open AND volume > 100000",
            "date >= '2023-01-01'"
          ]
        }
      },
      "required": ["symbol"]
    }

    Args:
        symbol: 基金名称, 可以通过 ak.fund_etf_category_sina() 函数获取
        condition: 筛选条件字符串，根据返回的字段组装类似于SQL的where过滤条件，格式示例:
           "close > open AND volume > 100000"
    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量

    Examples:
        ```python
        await fund_etf_hist_sina(
            symbol="sh510050",
            condition="close > open AND volume > 100000"
        )
        ```

    Error Handling:
        - 如果基金代码无效，返回None
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
    """
    df = ak.fund_etf_hist_sina(symbol)
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()



@mcp.tool()
async def fund_hk_rank_em(condition: str = None) -> pd.DataFrame:
    """
    东方财富网-数据中心-香港基金排行

    Input Schema:
    {
      "type": "object",
      "properties": {
        "condition": {
          "type": "string",
          "description": "筛选条件字符串，支持类似SQL的WHERE语法（仅限使用返回字段进行过滤）",
          "examples": [
            "日增长率 > 0 AND 近1年 > 10",
            "基金简称包含('中国')"
          ]
        }
      },
      "required": []
    }

    Returns:
        返回包含以下字段的DataFrame的Markdown格式:
        - 序号: 序号
        - 基金代码: 基金代码
        - 基金简称: 基金简称
        - 币种: 计价币种
        - 日期: 净值日期
        - 单位净值: 单位净值
        - 日增长率: 日增长率(%)
        - 近1周: 近1周收益率(%)
        - 近1月: 近1月收益率(%)
        - 近3月: 近3月收益率(%)
        - 近6月: 近6月收益率(%)
        - 近1年: 近1年收益率(%)
        - 近2年: 近2年收益率(%)
        - 近3年: 近3年收益率(%)
        - 今年来: 年初至今收益率(%)
        - 成立来: 成立以来总收益率(%)
        - 可购买: 是否可购买
        - 香港基金代码: 香港市场基金代码

    Examples:
        ```python
        await fund_hk_rank_em(
            condition="日增长率 > 0 AND 近1年 > 10"
        )
        ```

    Error Handling:
        - 如果条件语法错误，抛出ValueError
        - 如果数据获取失败，抛出RuntimeError
        - 如果使用未返回的字段进行过滤，抛出ValueError并提示可用字段
    """
    df = ak.fund_hk_rank_em()
    if condition:
        return apply_filters_for_data_frame(df, condition).to_markdown()
    return df.to_markdown()





# @mcp.tool()
# async def stock_news_em(symbol: str = "300059") -> pd.DataFrame:
#     """
#     东方财富-个股新闻-最近 100 条新闻
#     Args:
#         symbol: 股票代码

#     Returns:东方财富-个股新闻-最近 100 条新闻

#     """
#     return ak.stock_news_em(symbol).to_markdown()


# @mcp.tool()
# async def news_cctv(date: str = "20240424") -> pd.DataFrame:
#     """
#     新闻联播文字稿
#     Args:
#         date: 需要获取数据的日期; 目前 20160203 年后

#     Returns:新闻联播文字稿

#     """
#     return ak.news_cctv(date).to_markdown()




if __name__ == "__main__":
    
    # 启动MCP服务
    mcp.run()
