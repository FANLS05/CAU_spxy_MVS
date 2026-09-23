#!/usr/bin/env bash
# 断点续传下载器:网络反复断连时,断了就接着下,直到文件字节数达标
# 用法: ./fetch.sh <url> <输出文件> <显示名>

url="$1"; out="$2"; name="$3"

expect=$(curl -sIL --max-time 40 "$url" 2>/dev/null \
         | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')

if [[ -z "$expect" || "$expect" -le 0 ]] 2>/dev/null; then
    echo "$name: 无法获取文件大小,放弃"
    exit 1
fi

echo "$name: 期望 $((expect / 1048576)) MB (${expect} bytes)"

for i in $(seq 1 80); do
    cur=0
    [[ -f "$out" ]] && cur=$(stat -c%s "$out" 2>/dev/null || echo 0)

    if [[ "$cur" -ge "$expect" ]]; then
        echo "$name: ✓ 完整 ($((cur / 1048576)) MB,共 $i 次尝试)"
        exit 0
    fi

    printf '  [%2d] 已有 %6.1f / %6.1f MB,续传中...\n' \
        "$i" "$(echo "$cur/1048576" | bc -l 2>/dev/null || echo 0)" "$(echo "$expect/1048576" | bc -l 2>/dev/null || echo 0)"

    # --speed-limit/--speed-time: 低于 1KB/s 持续 60 秒就断开重来
    curl -sSL -C - \
         --connect-timeout 20 \
         --max-time 900 \
         --speed-limit 1024 --speed-time 60 \
         -o "$out" "$url" 2>/dev/null

    sleep 2
done

echo "$name: ✗ 超过最大尝试次数,仍未下完"
exit 1
