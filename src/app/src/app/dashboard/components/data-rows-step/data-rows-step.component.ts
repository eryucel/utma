import {AfterViewInit, Component, OnInit, ViewChild} from '@angular/core';
import {MatTableDataSource} from "@angular/material/table";
import {SendDatasetService} from "../../../core";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {SelectionModel} from "@angular/cdk/collections";

@Component({
  selector: 'app-data-rows-step',
  templateUrl: './data-rows-step.component.html',
  styleUrls: ['./data-rows-step.component.css']
})
export class DataRowsStepComponent implements OnInit, AfterViewInit {

  rowsDataSource: MatTableDataSource<any> = new MatTableDataSource<any>();
  displayedColumns: string[] = [];
  showMissing = false;
  selection = new SelectionModel<any>(false, []);
  // @ts-ignore
  @ViewChild(MatPaginator) paginator: MatPaginator;
  // @ts-ignore
  @ViewChild(MatSort) sort: MatSort;

  constructor(private sendDataset: SendDatasetService) {
  }

  ngOnInit(): void {
    this.rowsDataSource = new MatTableDataSource<any>(this.sendDataset.datasetDetail.rowsData.data);
    this.displayedColumns = this.sendDataset.datasetDetail.rowsData.meta.fields
  }

  ngAfterViewInit() {
    this.rowsDataSource.paginator = this.paginator;
    this.rowsDataSource.sort = this.sort;
  }

  applyFilter(event: Event): void {
    const filterValue = (event.target as HTMLInputElement).value;
    this.rowsDataSource.filter = filterValue.trim().toLowerCase();

    if (this.rowsDataSource.paginator) {
      this.rowsDataSource.paginator.firstPage();
    }
  }

  setDataSource(val: any[] | undefined): void {
    this.rowsDataSource = new MatTableDataSource<any>(val);
    this.rowsDataSource.paginator = this.paginator;
    this.rowsDataSource.sort = this.sort;
  }

  checkboxLabel(row?: any): string {
    return `${this.selection.isSelected(row) ? 'deselect' : 'select'} row ${row.position + 1}`;
  }

  toggleMissing(): void {
    this.showMissing = !this.showMissing;
    if (this.showMissing) {
      const tmpData: any[] = [];
      this.sendDataset.datasetDetail.rowsData.data.forEach((el: { [s: string]: unknown; } | ArrayLike<unknown>) => {
        if (Object.values(el).includes(null)) {
          tmpData.push(el);
        }
      });
      this.setDataSource(tmpData);
    } else {
      this.setDataSource(this.sendDataset.datasetDetail.rowsData.data);
    }
  }
}
