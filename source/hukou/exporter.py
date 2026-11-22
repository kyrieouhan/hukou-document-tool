"""Plain, useful Excel output with explicit text cells and atomic replacement."""
from pathlib import Path
import os,tempfile
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from .models import Household

HEADERS=['户号','户主','住址','成员姓名','与户主关系','身份证号码','登记日期','家庭总人口']
LABELS={'household_no':'户号','householder':'户主','address':'住址','name':'成员姓名','relation':'与户主关系',
        'id_number':'身份证号码','registration_date':'登记日期','family_size':'家庭总人口','members':'成员','page':'页面','orientation':'方向'}

def append_safe(sheet,values: list) -> None:
    sheet.append(values)
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value,str): cell.data_type='s'

def export_excel(households: list[Household],path: Path) -> None:
    book=Workbook(); summary=book.active; summary.title='家庭成员汇总'
    review=book.create_sheet('待复核'); log=book.create_sheet('处理日志')
    append_safe(summary,HEADERS)
    append_safe(review,['原PDF','页码','户主','成员姓名','异常字段','异常原因','当前识别值'])
    append_safe(log,['原文件名','新文件名','总页数','识别成员数','状态','处理时间','错误信息'])
    for h in households:
        for m in h.members:
            append_safe(summary,[h.household_no,h.householder,h.address,m.name,m.relation,m.id_number,m.registration_date,h.family_size])
            summary.cell(summary.max_row,6).number_format='@'
            summary.cell(summary.max_row,1).number_format='@'
            summary.cell(summary.max_row,7).number_format='@'
        for issue in h.issues:
            append_safe(review,[Path(h.source_pdf).name,issue.page or '',h.householder,issue.member,
                                LABELS.get(issue.field,issue.field),issue.reason,issue.value])
        append_safe(log,[Path(h.source_pdf).name,Path(h.new_pdf).name if h.new_pdf else '',h.page_count,len(h.members),
                         '成功' if h.status=='ok' else '失败' if h.status=='error' else '待复核',round(h.elapsed_seconds,2),
                         '；'.join(dict.fromkeys(i.reason for i in h.issues))])
    widths=[[18,14,56,14,25,25,16,16],[25,8,14,14,18,60,40],[25,25,12,14,12,14,60]]
    for sheet,sizes in zip(book,widths):
        sheet.freeze_panes='A2'; sheet.auto_filter.ref=sheet.dimensions
        for idx,width in enumerate(sizes,1):
            from openpyxl.utils import get_column_letter
            sheet.column_dimensions[get_column_letter(idx)].width=width
        for cell in sheet[1]:
            cell.font=Font(name='Microsoft YaHei',bold=True)
            cell.fill=PatternFill('solid',fgColor='E5EDF4')
        for row in sheet.iter_rows(min_row=2):
            for cell in row: cell.alignment=Alignment(vertical='top',wrap_text=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(suffix='.xlsx',dir=path.parent); os.close(fd)
    try:
        book.save(name); os.replace(name,path)
    finally:
        book.close()
        if os.path.exists(name): os.unlink(name)
