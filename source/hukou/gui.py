"""Qt front end; a background coordinator owns isolated recognition processes."""
import sys,json,time
from pathlib import Path
from PySide6.QtCore import QThread,Signal,QTimer,QUrl,Qt
from PySide6.QtGui import QDesktopServices,QFont
from PySide6.QtWidgets import (QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QLineEdit,
                             QPushButton,QFileDialog,QComboBox,QProgressBar,QCheckBox,QMessageBox,QScrollArea,QFrame)
from .batch import batch,scan_pdfs,Control
from .runtime import hardware_snapshot,choose_plan,GIB

class Worker(QThread):
    progress=Signal(dict)
    completed=Signal(dict)
    failed=Signal(str)
    def __init__(self,folder,output,naming,force,control,mode='auto',workers=None):
        super().__init__(); self.args=(folder,output,naming,force); self.control=control
        self.mode=mode;self.workers=workers
    def run(self):
        try: self.completed.emit(batch(*self.args,control=self.control,progress=self.progress.emit,
                                      mode=self.mode,workers=self.workers))
        except Exception as exc: self.failed.emit(str(exc) if isinstance(exc,(ValueError,RuntimeError)) else '处理失败：'+type(exc).__name__)

class Window(QWidget):
    def __init__(self):
        super().__init__(); self.worker=None; self.control=None; self.result={}; self.closing=False; self.last_progress={}
        self.setWindowTitle('户口资料批量整理工具 1.2.0'); self.setMinimumSize(680,480)
        available=QApplication.primaryScreen().availableGeometry()
        self.resize(min(850,available.width()-40),min(740,available.height()-60))
        outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.Shape.NoFrame)
        content=QWidget();scroll.setWidget(content);outer.addWidget(scroll)
        layout=QVBoxLayout(content); layout.setContentsMargins(28,24,28,24); layout.setSpacing(14)
        title=QLabel('户口资料批量整理工具'); title.setFont(QFont('Microsoft YaHei',20,QFont.Weight.Bold))
        layout.addWidget(title)
        intro=QLabel('本地离线识别 · 一份 PDF 对应一户 · 原始文件保持不动')
        intro.setStyleSheet('color:#516272;'); layout.addWidget(intro)
        self.input=QLineEdit(); self.input.setPlaceholderText('请选择村民小组文件夹')
        self.output=QLineEdit(); self.output.setPlaceholderText('默认保存在输入文件夹内的“输出”目录')
        self.choose_input=QPushButton('选择文件夹'); self.choose_output=QPushButton('选择目录')
        for label,edit,button in [('村民小组文件夹',self.input,self.choose_input),('输出目录',self.output,self.choose_output)]:
            layout.addWidget(QLabel(label)); row=QHBoxLayout(); row.addWidget(edit,1); row.addWidget(button); layout.addLayout(row)
        self.count=QLabel('检测到 0 个 PDF'); layout.addWidget(self.count)
        row=QHBoxLayout(); row.addWidget(QLabel('副本命名'))
        self.naming=QComboBox(); self.naming.addItems(['户主姓名','户主姓名_户号']); row.addWidget(self.naming)
        row.addStretch(); self.force=QCheckBox('重新处理全部'); row.addWidget(self.force); layout.addLayout(row)
        row=QHBoxLayout();row.addWidget(QLabel('性能模式'))
        self.performance=QComboBox()
        for label,key in [('自动（推荐）','auto'),('低资源','low'),('标准','standard'),('高性能','high'),('自定义','custom')]:
            self.performance.addItem(label,key)
        row.addWidget(self.performance);row.addStretch();row.addWidget(QLabel('同时处理户数'))
        self.parallel=QComboBox();self.parallel.addItem('自动',None)
        for n in range(1,5):self.parallel.addItem(str(n),n)
        row.addWidget(self.parallel);layout.addLayout(row)
        self.hardware=QLabel();self.hardware.setWordWrap(True);self.hardware.setStyleSheet('color:#516272;')
        layout.addWidget(self.hardware)
        self.performance.currentIndexChanged.connect(self.refresh_hardware)
        self.parallel.currentIndexChanged.connect(self.refresh_hardware)
        self.refresh_hardware()
        row=QHBoxLayout(); self.start=QPushButton('开始识别'); self.pause=QPushButton('暂停'); self.stop=QPushButton('停止')
        self.start.setStyleSheet('background:#215c87;color:white;font-weight:bold;')
        for button in (self.start,self.pause,self.stop): button.setMinimumHeight(38); row.addWidget(button)
        layout.addLayout(row); self.pause.setEnabled(False); self.stop.setEnabled(False)
        self.current=QLabel('准备就绪'); self.current.setWordWrap(True); layout.addWidget(self.current)
        self.bar=QProgressBar(); self.bar.setValue(0);self.bar.setFormat('尚未开始'); layout.addWidget(self.bar)
        self.stats=QLabel('自动通过 0 户    待复核 0 户    待检查 0 项'); layout.addWidget(self.stats)
        self.speed=QLabel('平均处理速度与剩余时间：正在估算');layout.addWidget(self.speed)
        note=QLabel('请先查看 Excel 的“待复核”页。家庭人口留空表示尚未可靠确认。')
        note.setWordWrap(True); note.setStyleSheet('color:#516272;'); layout.addWidget(note)
        row=QHBoxLayout(); self.open_excel=QPushButton('打开 Excel'); self.open_output=QPushButton('打开输出目录')
        row.addWidget(self.open_excel); row.addWidget(self.open_output); layout.addLayout(row)
        self.open_excel.setEnabled(False); self.open_output.setEnabled(False)
        self.choose_input.clicked.connect(self.select_input); self.choose_output.clicked.connect(self.select_output)
        self.input.editingFinished.connect(self.refresh_count); self.output.editingFinished.connect(self.refresh_count)
        self.start.clicked.connect(self.begin); self.pause.clicked.connect(self.toggle_pause); self.stop.clicked.connect(self.stop_work)
        self.open_excel.clicked.connect(lambda:self.open_path(self.result.get('excel','')))
        self.open_output.clicked.connect(lambda:self.open_path(self.result.get('output',self.output.text())))
        self.setStyleSheet(self.styleSheet()+'QLineEdit,QComboBox {padding:7px;border:1px solid #b9c6ce;border-radius:4px;} QPushButton {padding:8px 14px;}')
        self.pause_timer=QTimer(self); self.pause_timer.setInterval(200)
        self.pause_timer.timeout.connect(self.refresh_pause_state); self.pause_timer.start()

    def select_input(self):
        selected=QFileDialog.getExistingDirectory(self,'选择村民小组文件夹',self.input.text())
        if selected:
            self.input.setText(selected); self.output.setText(str(Path(selected)/'输出')); self.refresh_count()
    def select_output(self):
        selected=QFileDialog.getExistingDirectory(self,'选择输出目录',self.output.text())
        if selected: self.output.setText(selected); self.refresh_count()
    def refresh_count(self):
        try:
            folder=Path(self.input.text()); output=Path(self.output.text()) if self.output.text() else folder/'输出'
            self.count.setText(f'检测到 {len(scan_pdfs(folder,output))} 个 PDF（只读取此文件夹，不包含子文件夹）')
        except ValueError: self.count.setText('请选择有效的输入文件夹')
    def refresh_hardware(self):
        hw=hardware_snapshot();plan=choose_plan(self.performance.currentData(),self.parallel.currentData(),hw)
        self.parallel.setEnabled(self.performance.currentData()!='low' and not (self.worker and self.worker.isRunning()))
        self.hardware.setText(f"检测到 {hw['physical_cpus']} 核 / {hw['logical_cpus']} 线程，内存 {hw['total_memory']/GIB:.0f} GB；"
                              f"建议同时处理 {plan.recommended} 户。"+(' '+plan.warning if plan.warning else ''))
    def begin(self):
        if self.worker and self.worker.isRunning(): return
        if not self.input.text().strip(): self.current.setText('请先选择村民小组文件夹'); return
        folder=Path(self.input.text()); output=Path(self.output.text()) if self.output.text() else folder/'输出'
        self.output.setText(str(output)); self.control=Control(); self.result={}; self.last_progress={}
        self.stats.setText('自动通过 0 户    待复核 0 户    待检查 0 项')
        self.speed.setText('平均处理速度与剩余时间：正在估算')
        self.worker=Worker(folder,output,'name_number' if self.naming.currentIndex() else 'name',self.force.isChecked(),self.control,
                           self.performance.currentData(),self.parallel.currentData())
        self.worker.progress.connect(self.update_progress); self.worker.completed.connect(self.done); self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.finished); self.set_busy(True); self.bar.setValue(0)
        self.current.setText('正在加载本地识别模型…'); self.worker.start()
    def set_busy(self,busy):
        for widget in (self.input,self.output,self.choose_input,self.choose_output,self.naming,self.force,self.start,self.performance,self.parallel): widget.setEnabled(not busy)
        self.pause.setEnabled(busy); self.stop.setEnabled(busy)
        if busy: self.open_excel.setEnabled(False); self.open_output.setEnabled(False)
    def toggle_pause(self):
        if not self.control:return
        if self.control.paused.is_set(): self.control.paused.clear(); self.pause.setText('暂停'); self.current.setText('正在继续识别…')
        else: self.control.paused.set(); self.pause.setText('继续'); self.current.setText('正在暂停领取新任务；当前户处理完成后会安全保存。')
    def refresh_pause_state(self):
        if self.control and self.control.paused.is_set() and self.control.pause_acknowledged.is_set():
            self.current.setText('已暂停领取新 PDF；正在处理的户会继续保存。点击“继续”恢复领取。')
    def stop_work(self):
        if self.control: self.control.stopped.set(); self.current.setText('正在停止，当前识别步骤结束后生效…')
    def update_progress(self,value):
        self.last_progress=value
        self.bar.setFormat('%v / %m 户')
        self.bar.setMaximum(value['total']); self.bar.setValue(value['index'])
        page=f" · 第 {value['page']}/{value['pages']} 页" if 'page' in value else ''
        self.current.setText(value.get('stage','')+' · '+value['file']+page)
        if 'ok' in value: self.stats.setText(f"自动通过 {value['ok']} 户    待复核 {value['review']} 户    待检查 {value.get('review_fields',0)} 项")
        workers=value.get('workers',1);average=value.get('average_seconds');eta=value.get('eta_seconds')
        speed=f'约 {average/workers:.1f} 秒/户' if average else '正在估算'
        remaining=f'约 {max(1,round(eta/60))} 分钟' if eta is not None else '正在估算'
        self.speed.setText(f'同时处理 {workers} 户    平均速度：{speed}    预计剩余：{remaining}')
        if value.get('plan',{}).get('warning'):self.hardware.setText(value['plan']['warning'])
    def done(self,result):
        self.result=result
        self.speed.setText(f"本次用时 {result.get('seconds',0)/60:.1f} 分钟")
        self.stats.setText(f"自动通过 {result['ok']} 户    待复核 {result['review']} 户    待检查 {result.get('review_fields',0)} 项")
        self.current.setText(result.get('error') or ('已停止，下次开始将从断点继续。' if result['cancelled'] else f"处理完成，共 {result['processed']} 户，其中恢复 {result['resumed']} 户。"))
        self.open_excel.setEnabled(Path(result['excel']).is_file()); self.open_output.setEnabled(True)
    def error(self,message): self.current.setText(message)
    def finished(self):
        self.set_busy(False); self.pause.setText('暂停')
        self.refresh_hardware()
        if self.closing: self.close()
    def open_path(self,value):
        if value and Path(value).exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(value).resolve())))
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.closing=True; self.stop_work(); event.ignore()
        else: event.accept()

def launch(smoke: Path|None=None) -> int:
    app=QApplication.instance() or QApplication(sys.argv[:1]); app.setFont(QFont('Microsoft YaHei',10))
    window=Window(); window.show()
    if smoke:
        from PySide6.QtTest import QTest
        config=json.loads(smoke.read_text(encoding='utf-8')); report=Path(config['report']); report.parent.mkdir(parents=True,exist_ok=True)
        window.input.setText(config['input']); window.output.setText(config['output']); window.refresh_count()
        window.performance.setCurrentIndex(max(0,window.performance.findData(config.get('mode','auto'))))
        window.parallel.setCurrentIndex(max(0,window.parallel.findData(config.get('workers'))))
        window.force.setChecked(config.get('force',False))
        beats=[0]; started=[False]; paused=[False]; acknowledged=[False]; began=time.perf_counter()
        last_beat=[began];heartbeat_gaps=[]
        last_reported=[None]
        timer=QTimer(); timer.setInterval(100)
        def tick():
            now=time.perf_counter();heartbeat_gaps.append(now-last_beat[0]);last_beat[0]=now
            if config.get('progress_report') and window.last_progress:
                value=window.last_progress
                marker=(value.get('index'),value.get('stage'),value.get('file'),value.get('page'))
                if marker!=last_reported[0]:
                    from .storage import atomic_json
                    atomic_json(Path(config['progress_report']),{k:value[k] for k in ('index','total','stage','ok','review','workers') if k in value})
                    last_reported[0]=marker
            beats[0]+=1
            if paused[0] and not acknowledged[0] and window.control and window.control.pause_acknowledged.is_set():
                acknowledged[0]=True
                window.grab().save(str(report.with_name(report.stem+'-paused.png')))
                QTimer.singleShot(config.get('pause_duration_ms',500),lambda:QTest.mouseClick(window.pause,Qt.MouseButton.LeftButton))
            if not started[0]:
                window.grab().save(str(report.with_suffix('.png'))); QTest.mouseClick(window.start,Qt.MouseButton.LeftButton); started[0]=True
            elif window.worker and window.worker.isRunning() and config.get('stop_after') and window.last_progress.get('index',0)>=config['stop_after']:
                QTest.mouseClick(window.stop,Qt.MouseButton.LeftButton)
            elif window.worker and window.worker.isRunning() and config.get('test_pause',True) and not paused[0] and (window.last_progress.get('index',0)>=1 if config.get('pause_on_save') else beats[0]>20):
                QTest.mouseClick(window.pause,Qt.MouseButton.LeftButton); paused[0]=True
            elif window.worker and not window.worker.isRunning():
                data=dict(window.result,heartbeat_count=beats[0],pause_tested=paused[0],pause_acknowledged=acknowledged[0],elapsed=time.perf_counter()-began,
                          max_heartbeat_gap_seconds=max(heartbeat_gaps),
                          p95_heartbeat_gap_seconds=sorted(heartbeat_gaps)[int((len(heartbeat_gaps)-1)*.95)],
                          stats_text=window.stats.text(),
                          status_text=window.current.text(),excel_button_enabled=window.open_excel.isEnabled())
                report.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                window.grab().save(str(report.with_name(report.stem+'-finished.png')))
                timer.stop(); app.quit()
        timer.timeout.connect(tick); timer.start()
    return app.exec()
