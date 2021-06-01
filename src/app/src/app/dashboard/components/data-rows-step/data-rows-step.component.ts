import {AfterViewInit, Component, OnInit, ViewChild} from '@angular/core';
import {MatTableDataSource} from "@angular/material/table";
import {SendDatasetService} from "../../../core";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {SelectionModel} from "@angular/cdk/collections";
import {DatasetService} from "../../../core/services/dataset.service";
import {FileService} from "../../../core/services/file.service";
import {Router} from "@angular/router";

@Component({
  selector: 'app-data-rows-step',
  templateUrl: './data-rows-step.component.html',
  styleUrls: ['./data-rows-step.component.css'],
  providers: [DatasetService, FileService]
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

  constructor(private sendDataset: SendDatasetService,
              private datasetService: DatasetService, private router: Router,
              private fileService: FileService) {
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

  save(): void {
    const replacer = (key: any, value: null) => (value === null ? '' : value);
    const header = Object.keys(this.sendDataset.datasetDetail.rowsData.data[0]);
    const csv = this.sendDataset.datasetDetail.rowsData.data.map((row: { [x: string]: any; }) =>
      header
        .map((fieldName) => JSON.stringify(row[fieldName], replacer))
        .join(',')
    );
    csv.unshift(header.join(','));
    const csvArray = csv.join('\r\n');

    const file = new File([csvArray], this.sendDataset.datasetDetail.name.replace(' ', '_') + '.csv', {type: 'text/csv'})

    this.fileService.postFile(this.sendDataset.datasetDetail.name, file).subscribe(result => {
      const splitted = result.file.split("/");
      let categoricalAttributes = "";
      let numericAttributes = "";
      this.sendDataset.datasetDetail.categoricalAttributes?.map(attr => categoricalAttributes += attr.name + ",");
      this.sendDataset.datasetDetail.numberAttributes?.map(attr => numericAttributes += attr.name + ",");
      if (this.sendDataset.datasetDetail.id == 0) {
        this.datasetService.postDataset({
          id: undefined,
          data: splitted[splitted.length - 1],
          name: this.sendDataset.datasetDetail.name,
          col: this.sendDataset.datasetDetail.rowsData.meta.fields.length,
          row: this.sendDataset.datasetDetail.rowsData.data.length,
          categoricalAttributes: categoricalAttributes.slice(0, -1),
          numericAttributes: numericAttributes.slice(0, -1)
        }).subscribe();
      } else {
        this.datasetService.updateDataset({
          id: this.sendDataset.datasetDetail.id,
          data: splitted[splitted.length - 1],
          name: this.sendDataset.datasetDetail.name,
          // @ts-ignore
          col: this.sendDataset.datasetDetail.categoricalAttributes?.length + this.sendDataset.datasetDetail.numberAttributes?.length,
          row: this.sendDataset.datasetDetail.rowsData.data.length,
          categoricalAttributes: categoricalAttributes.slice(0, -1),
          numericAttributes: numericAttributes.slice(0, -1)
        }).subscribe();
      }
      this.router.navigate(['/dashboard/datasets']);
    });
  }
}
