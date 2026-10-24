<template>
  <div class="screen">
    <header>
      <h1>SOP-AI 数据大屏 <small>本地训推一体 Demo · 一期</small></h1>
      <div class="clock">{{ clock }}</div>
    </header>

    <section class="cards">
      <div class="card"><div class="num">{{ summary.runs_total ?? '–' }}</div><div class="label">累计运行次数</div></div>
      <div class="card">
        <div class="num">{{ lastTrainMetric('mAP50(P)') ?? '–' }}</div>
        <div class="label">最新训练 姿态 mAP@50</div>
      </div>
      <div class="card">
        <div class="num">{{ lastTrainMetric('mAP50(B)') ?? '–' }}</div>
        <div class="label">最新训练 框 mAP@50</div>
      </div>
      <div class="card"><div class="num">{{ backendOk ? '在线' : '离线' }}</div><div class="label">核心服务</div></div>
    </section>

    <section class="panels">
      <div class="panel">
        <h2>最近运行（runs/ 自动归档）</h2>
        <ul>
          <li v-for="r in summary.runs ?? []" :key="r">
            <span>{{ r }}</span>
          </li>
        </ul>
      </div>
      <div class="panel">
        <h2>实时检测事件（events.jsonl）</h2>
        <ul>
          <li v-for="(e, i) in summary.events ?? []" :key="i">
            frame {{ e.frame }} · {{ e.fps }} fps · {{ e.persons }} 人
          </li>
          <li v-if="!(summary.events ?? []).length" class="dim">暂无事件 —— 运行 src/main.py run --video 后出现</li>
        </ul>
      </div>
      <div class="panel">
        <h2>训练指标趋势（results.csv）</h2>
        <div ref="chartEl" class="chart"></div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, nextTick } from 'vue'
import * as echarts from 'echarts'

const summary = ref({})
const clock = ref('')
const chartEl = ref(null)
const backendOk = ref(false)
let timer = null
let chart = null

function lastTrainMetric(key) {
  const m = summary.value.latest_train?.final_metrics ?? {}
  const hit = Object.keys(m).find((k) => k.endsWith(key))
  return hit ? Number(m[hit]).toFixed(4) : null
}

async function refresh() {
  try {
    const r = await fetch('/api/summary')
    if (r.ok) {
      summary.value = await r.json()
      backendOk.value = true
      renderChart()
    } else {
      backendOk.value = false
    }
  } catch {
    backendOk.value = false
  }
}

async function renderChart() {
  const t = summary.value.latest_train
  if (!t || !chartEl.value) return
  try {
    const r = await fetch(`/api/train_metrics?run=${encodeURIComponent(t.run)}`)
    if (!r.ok) return
    const data = await r.json()
    await nextTick()
    if (!chart) chart = echarts.init(chartEl.value)
    chart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: data.series.map((s) => s.name), textStyle: { color: '#9fb3c8' } },
      xAxis: { type: 'category', data: data.epochs, textStyle: { color: '#9fb3c8' } },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: '#1d2b3a' } } },
      series: data.series,
    })
  } catch { /* chart optional */ }
}

onMounted(() => {
  refresh()
  timer = setInterval(() => {
    clock.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    refresh()
  }, 3000)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<style>
* { box-sizing: border-box; margin: 0; }
body { background: #0b1420; color: #dce8f5; font-family: 'Microsoft YaHei', system-ui, sans-serif; }
.screen { padding: 18px 22px; height: 100vh; display: flex; flex-direction: column; gap: 14px; }
header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #1d2b3a; padding-bottom: 10px; }
header h1 { font-size: 22px; }
header small { color: #6f8ba4; font-size: 13px; margin-left: 10px; }
.clock { font-variant-numeric: tabular-nums; color: #6f8ba4; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.card { background: #101d2c; border: 1px solid #1d2b3a; border-radius: 10px; padding: 14px 18px; }
.card .num { font-size: 30px; font-weight: 700; color: #4dd0a6; }
.card .label { color: #6f8ba4; font-size: 13px; margin-top: 4px; }
.panels { display: grid; grid-template-columns: 1fr 1fr 1.4fr; gap: 14px; flex: 1; min-height: 0; }
.panel { background: #101d2c; border: 1px solid #1d2b3a; border-radius: 10px; padding: 14px 16px; overflow: auto; }
.panel h2 { font-size: 15px; color: #9fb3c8; margin-bottom: 10px; }
.panel ul { list-style: none; font-size: 13px; line-height: 1.9; }
.panel .dim { color: #51667c; }
.chart { height: 320px; }
</style>
