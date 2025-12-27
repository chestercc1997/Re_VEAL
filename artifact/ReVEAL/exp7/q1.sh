#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DIR="../64_default"
DC2_DIR="../64_dc2"
ABC="../../../exp_tool/abc/abc"

OUT_CSV="and_compare.csv"
LOG_FILE="and_compare.log"

echo "index,default_and,dc2_and,diff,rel_diff(% )" > "$OUT_CSV"
: > "$LOG_FILE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

index=0
sum_diff=0              # sum(dc2 - default)
sum_diff_sq=0           # sum((dc2 - default)^2)
sum_rel=0               # sum((dc2 - default)/default)
sum_rel_sq=0            # sum(((dc2 - default)/default)^2)
count=0                 # number of valid pairs with default>0
sum_default=0
sum_dc2=0

extract_and_strict() {
  perl -0777 -ne '
    my $t = $_;
    my @cands;

    while ($t =~ /(?im)^\s*(?:and(?:\s*nodes)?|number of and nodes)\s*[:=]\s*([0-9]+)\s*$/g) {
      push @cands, $1;
    }
    while ($t =~ /(?i)\bAND\s*(?:nodes)?\s*[:=]\s*([0-9]+)/g) {
      push @cands, $1;
    }
    if (@cands) { print $cands[-1], "\n"; exit; }

    if ($t =~ /(?i)\band\b[^\d]{0,10}([0-9]+)/) { print $1, "\n"; exit; }
  '
}

get_and_count() {
  local file="$1"
  [[ -f "$file" ]] || { echo "ERR"; return 0; }

  {
    echo "[$(timestamp)] ---- BEGIN ABC ----"
    echo "[$(timestamp)] File: $file"
    echo "[$(timestamp)] Cmd:  $ABC -c \"read_aiger \\\"$file\\\"; ps\""
  } >> "$LOG_FILE"

  local out
  if ! out=$("$ABC" -c "read_aiger \"$file\"; ps" 2>&1); then
    {
      echo "[$(timestamp)] ABC returned non-zero for $file"
      echo "[$(timestamp)] --- ABC OUTPUT START ---"
      printf "%s\n" "$out"
      echo "[$(timestamp)] --- ABC OUTPUT END ---"
      echo "[$(timestamp)] ---- END ABC (ERROR) ----"
    } >> "$LOG_FILE"
    echo "ERR"
    return 0
  else
    {
      echo "[$(timestamp)] --- ABC OUTPUT START ---"
      printf "%s\n" "$out"
      echo "[$(timestamp)] --- ABC OUTPUT END ---"
      echo "[$(timestamp)] ---- END ABC (OK) ----"
    } >> "$LOG_FILE"
  fi

  local andv
  andv=$(printf "%s\n" "$out" | extract_and_strict || true)
  if [[ -z "$andv" ]]; then
    echo "[$(timestamp)] [PARSE ERR] Failed to extract AND for $file" >> "$LOG_FILE"
    echo "ERR"
  else
    echo "[$(timestamp)] Parsed AND: $andv" >> "$LOG_FILE"
    echo "$andv"
  fi
}

shopt -s nullglob
mapfile -t default_files < <(find "$DEFAULT_DIR" -maxdepth 1 -type f -name "*_default.aig" | sort)

{
  echo "[$(timestamp)] ===== Start compare run ====="
  echo "Default dir: $DEFAULT_DIR"
  echo "DC2 dir:     $DC2_DIR"
  echo "ABC path:    $ABC"
  echo "Total default candidates: ${#default_files[@]}"
} >> "$LOG_FILE"

for def_file in "${default_files[@]}"; do
  base="$(basename "$def_file")"
  dc2_base="${base%_default.aig}_dc2.aig"
  dc2_file="$DC2_DIR/$dc2_base"

  {
    echo "[$(timestamp)] ---- Pair ----"
    echo "Default: $def_file"
    echo "DC2:     $dc2_file"
  } >> "$LOG_FILE"

  if [[ ! -f "$dc2_file" ]]; then
    echo "[$(timestamp)] [MISS] $dc2_base" >> "$LOG_FILE"
    continue
  fi

  def_and=$(get_and_count "$def_file")
  [[ "$def_and" == "ERR" ]] && { echo "[$(timestamp)] Skip pair due to default parse error." >> "$LOG_FILE"; continue; }

  dc2_and=$(get_and_count "$dc2_file")
  [[ "$dc2_and" == "ERR" ]] && { echo "[$(timestamp)] Skip pair due to dc2 parse error." >> "$LOG_FILE"; continue; }

  diff=$(( dc2_and - def_and ))

  if [[ "$def_and" -eq 0 ]]; then
    rel="NA"
    rel_pct="NA"
  else
    rel=$(awk -v d="$diff" -v b="$def_and" 'BEGIN{printf "%.8f", d*1.0/b}')
    rel_pct=$(awk -v r="$rel" 'BEGIN{printf "%.4f", r*100.0}')
  fi

  index=$((index + 1))
  echo "$index,$def_and,$dc2_and,$diff,$rel_pct" >> "$OUT_CSV"

  sum_diff=$(( sum_diff + diff ))
  sum_diff_sq=$(( sum_diff_sq + diff*diff ))
  sum_default=$(( sum_default + def_and ))
  sum_dc2=$(( sum_dc2 + dc2_and ))

  if [[ "$def_and" -ne 0 ]]; then
    #  awk  bash 
    sum_rel=$(awk -v a="$sum_rel" -v r="$rel" 'BEGIN{printf "%.12f", a + r}')
    sum_rel_sq=$(awk -v a="$sum_rel_sq" -v r="$rel" 'BEGIN{printf "%.12f", a + r*r}')
    count=$(( count + 1 ))
  else
    echo "[$(timestamp)] [WARN] default_and == 0, skip relative stats for this pair." >> "$LOG_FILE"
  fi

  echo "[$(timestamp)] Result row: index=$index, default=$def_and, dc2=$dc2_and, diff=$diff, rel%=$rel_pct" >> "$LOG_FILE"
done

echo "total,$sum_default,$sum_dc2,$sum_diff," >> "$OUT_CSV"

if [[ "$count" -gt 0 ]]; then
  avg_diff=$(awk -v s="$sum_diff" -v n="$index" 'BEGIN{printf "%.6f", s*1.0/n}')
  mse=$(awk -v s2="$sum_diff_sq" -v n="$index" 'BEGIN{printf "%.6f", s2*1.0/n}')

  avg_rel=$(awk -v s="$sum_rel" -v n="$count" 'BEGIN{printf "%.8f", s/n}')
  mse_rel=$(awk -v s="$sum_rel_sq" -v n="$count" 'BEGIN{printf "%.8f", s/n}')
  avg_rel_pct=$(awk -v r="$avg_rel" 'BEGIN{printf "%.4f", r*100.0}')
  mse_rel_pct=$(awk -v r="$mse_rel" 'BEGIN{printf "%.4f", r*100.0}')

  overall_ratio=$(awk -v sdc2="$sum_dc2" -v sdef="$sum_default" 'BEGIN{if(sdef==0) print "NA"; else printf "%.8f", sdc2/sdef}')
  overall_rel=$(awk -v r="$overall_ratio" 'BEGIN{if(r=="NA") print "NA"; else printf "%.8f", r-1.0}')
  overall_rel_pct=$(awk -v r="$overall_rel" 'BEGIN{if(r=="NA") print "NA"; else printf "%.4f", r*100.0}')
else
  avg_diff="NA"
  mse="NA"
  avg_rel="NA"
  mse_rel="NA"
  avg_rel_pct="NA"
  mse_rel_pct="NA"
  overall_ratio="NA"
  overall_rel="NA"
  overall_rel_pct="NA"
fi

#  CSV
echo "average_abs_diff,$avg_diff" >> "$OUT_CSV"
echo "mean_squared_abs_diff,$mse" >> "$OUT_CSV"
echo "average_rel_diff,,$avg_rel,,$avg_rel_pct%" >> "$OUT_CSV"
echo "mean_squared_rel_diff,,$mse_rel,,$mse_rel_pct%" >> "$OUT_CSV"
echo "overall_ratio(dc2/default),,$overall_ratio" >> "$OUT_CSV"
echo "overall_rel_diff,,$overall_rel,,$overall_rel_pct%" >> "$OUT_CSV"

{
  echo "[$(timestamp)] ===== Finished ====="
  echo "Pairs with relative stats: $count (skipped those with default=0)"
  echo "Totals: default=$sum_default, dc2=$sum_dc2, diff=$sum_diff"
  echo "Avg abs diff: $avg_diff"
  echo "MSE abs diff: $mse"
  echo "Avg rel diff: $avg_rel (=${avg_rel_pct}%)"
  echo "MSE rel diff: $mse_rel (=${mse_rel_pct}%)"
  echo "Overall ratio dc2/default: $overall_ratio"
  echo "Overall rel diff: $overall_rel (=${overall_rel_pct}%)"
} >> "$LOG_FILE"

echo "Done. CSV -> $OUT_CSV; Log -> $LOG_FILE"