import streamlit as st
import pandas as pd
import numpy as np
import re
import io

# --- 住所データの正規化関数 ---
def normalize_address(address):
    if pd.isna(address):
        return ""
    return str(address).replace('　', ' ').strip()

# --- 住所の厳密な正規化関数 (オプション) ---
def rigorous_normalize_address(address):
    if pd.isna(address):
        return ""
    addr = str(address).replace('　', ' ').strip()
    addr = re.sub(r'([都道府県])', r'\1 ', addr)
    addr = re.sub(r'([市区町村])', r'\1 ', addr)
    addr = re.sub(r'丁目', '-', addr)
    addr = re.sub(r'番地', '-', addr)
    addr = re.sub(r'番', '-', addr)
    addr = re.sub(r'号', '', addr)
    addr = re.sub(r'-+', '-', addr)
    addr = re.sub(r'-\s*$', '', addr)
    addr = re.sub(r'\s+', '', addr)
    return addr

# --- Streamlit UI ---
st.title("店舗ID統合ツール")
st.write("CSVファイルをアップロードし、「tacoms 対応ステータス」に基づいてCamelLocationIDを統合します。")

# ファイルアップローダー
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    st.success("ファイルが正常にアップロードされました。")

    # --- ファイルの読み込み ---
    try:
        df_original = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
    except UnicodeDecodeError:
        try:
            df_original = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('shift_jis')))
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました。エンコーディングを確認してください: {e}")
            df_original = None
    except Exception as e:
        st.error(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        df_original = None

    if df_original is not None:
        st.subheader("アップロードされたデータのプレビュー")
        st.dataframe(df_original.head())

        # カラム名の整形
        def clean_column_name(col_name):
            return col_name.replace('\n', ' ').replace('"', '').strip()
        
        df_original.columns = [clean_column_name(col) for col in df_original.columns]
        
        st.subheader("整形後のカラム名")
        st.write(df_original.columns.tolist())

        # カラム名定義 (固定値)
        address_col = 'StoreAddress'
        camel_id_source_col = 'CamelLocationID (Camel拠点ID)'
        tacoms_status_col = 'tacoms 対応ステータス'
        
        # 出力に含めるカラム名リスト（整形後の名前で指定）
        output_base_cols = [
            'z',
            'EnterpriseName（企業名） please write in Japanese.',
            'CamelLocationID (Camel拠点ID)',
            'CamelLocationName( Camel拠点名) please write in Japanese.',
            'RocketNow StoreID',
            'RocketNow StoreName',
            'StoreAddress'
        ]
        new_matched_id_col = 'Matched_CamelLocationID'
        output_cols = output_base_cols + [new_matched_id_col]


        # 必須カラムの存在チェック
        required_cols_for_processing = list(set([address_col, camel_id_source_col, tacoms_status_col] + output_base_cols))

        missing_cols = []
        for col in required_cols_for_processing:
            if col not in df_original.columns:
                missing_cols.append(col)

        if missing_cols:
            st.error(f"以下の必須カラムが見つかりません。CSVファイルのカラム名を確認してください: {', '.join(missing_cols)}")
        else:
            st.success("すべての必須カラムが存在します。")
            
            use_rigorous_normalization = st.checkbox("厳密な住所正規化を使用する（例: 丁目->-、スペース除去など）", value=False)
            
            if st.button("処理を開始"):
                with st.spinner("データ処理中..."):
                    if use_rigorous_normalization:
                        df_original[address_col + '_normalized'] = df_original[address_col].apply(rigorous_normalize_address)
                        st.info("厳密な住所正規化を適用しました。")
                    else:
                        df_original[address_col + '_normalized'] = df_original[address_col].apply(normalize_address)
                        st.info("基本的な住所正規化を適用しました。")

                    df_new = df_original[df_original[tacoms_status_col].isna() | (df_original[tacoms_status_col] == '')].copy()
                    df_past = df_original[df_original[tacoms_status_col].notna() & (df_original[tacoms_status_col] != '')].copy()

                    st.write(f"過去データ件数: {len(df_past)}")
                    st.write(f"新規データ件数: {len(df_new)}")

                    address_to_camel_id = {}
                    for index, row in df_past.iterrows():
                        normalized_addr = row[address_col + '_normalized']
                        camel_id = row[camel_id_source_col]
                        
                        if not normalized_addr or pd.isna(camel_id):
                            continue

                        if normalized_addr not in address_to_camel_id:
                            address_to_camel_id[normalized_addr] = camel_id
                            
                    df_new[new_matched_id_col] = np.nan

                    for index, row in df_new.iterrows():
                        normalized_addr = row[address_col + '_normalized']
                        if normalized_addr in address_to_camel_id:
                            df_new.at[index, new_matched_id_col] = address_to_camel_id[normalized_addr]

                    df_new_output_temp = df_new.drop(columns=[address_col + '_normalized'], errors='ignore')
                    
                    final_output_cols = [col for col in output_cols if col in df_new_output_temp.columns]
                    df_new_output = df_new_output_temp[final_output_cols]


                st.subheader("処理結果（新規データのみ）")
                st.dataframe(df_new_output.head(10))
                st.write(f"処理が完了しました。新規データ（{len(df_new_output)}件）のみが出力対象です。")

                # --- 変更点: UTF-8とShift-JISの2つのダウンロードボタンを生成 ---
                
                # UTF-8 (BOMあり)
                csv_output_utf8_bom = df_new_output.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="結果CSVをダウンロード (UTF-8 BOMあり)",
                    data=csv_output_utf8_bom,
                    file_name="新規データ_統合結果_CamelID_UTF8.csv",
                    mime="text/csv"
                )

                # Shift-JIS (CP932)
                try:
                    csv_output_sjis = df_new_output.to_csv(index=False, encoding='shift_jis')
                    st.download_button(
                        label="結果CSVをダウンロード (Shift-JIS)",
                        data=csv_output_sjis,
                        file_name="新規データ_統合結果_CamelID_SJIS.csv",
                        mime="text/csv"
                    )
                except UnicodeEncodeError as e:
                    st.warning(f"Shift-JISエンコーディングでエラーが発生しました。一部の文字が対応していません: {e}")
                    st.write("Shift-JISでのダウンロードはできませんでした。UTF-8をご利用ください。")

else:
    st.info("CSVファイルをアップロードして開始してください。")