import {AfterViewInit, Component, OnInit, ViewChild} from '@angular/core';
import {SelectionModel} from "@angular/cdk/collections";
import {MatTableDataSource} from "@angular/material/table";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {NgbModal} from "@ng-bootstrap/ng-bootstrap";
import {FormBuilder, FormGroup, Validators} from "@angular/forms";
import {
  AlertService,
  AttributeCategory,
  CategoricalAttribute,
  NumberAttribute,
  SendDatasetService
} from "../../../core";
import {Papa} from "ngx-papaparse";
import {Router} from "@angular/router";
import {FileService} from "../../../core/services/file.service";
import {DatasetService} from "../../../core/services/dataset.service";


@Component({
  selector: 'app-dataset',
  templateUrl: './dataset.component.html',
  styleUrls: ['./dataset.component.css']
})
export class DatasetComponent implements OnInit, AfterViewInit {

  selection = new SelectionModel<any>(false, []);
  displayedColumns: string[] = ['select', 'name', 'col', 'row'];
  dataSource: MatTableDataSource<any>;
  rowsData: any;
  fileFormGroup: FormGroup;
  numberAttributesData: NumberAttribute[] = [];
  categoricalAttributesData: CategoricalAttribute[] = [];

  @ViewChild(MatPaginator) paginator: MatPaginator | null = null;
  @ViewChild(MatSort) sort: MatSort | null = null;

  constructor(private papa: Papa,
              private modalService: NgbModal,
              private formBuilder: FormBuilder,
              private alertService: AlertService,
              private router: Router,
              private sendDataset: SendDatasetService,
              private fileService: FileService,
              private datasetService: DatasetService) {
    this.fileFormGroup = this.formBuilder.group({
      datasetName: ['', Validators.required],
      file: ['', Validators.required]
    });
    this.dataSource = new MatTableDataSource();
  }

  ngAfterViewInit(): void {
    this.dataSource.paginator = this.paginator;
    this.dataSource.sort = this.sort;
  }

  ngOnInit(): void {
    this.datasetService.getDatasets().subscribe(res => {
      this.dataSource = new MatTableDataSource(res);
      this.dataSource.paginator = this.paginator;
      this.dataSource.sort = this.sort;
    });
  }

  applyFilter(event: Event): void {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();

    if (this.dataSource.paginator) {
      this.dataSource.paginator.firstPage();
    }
  }

  checkboxLabel(row?: any): string {
    return `${this.selection.isSelected(row) ? 'deselect' : 'select'} row ${row.position + 1}`;
  }

  open(content: any) {
    this.modalService.open(content, {ariaLabelledBy: 'modal-basic-title'});
  }

  onSelect(event: any): void {
    if (this.fileFormGroup.controls.file) {
      this.onRemove();
    }
    this.fileFormGroup.controls.file.setValue(event.addedFiles[0]);
    this.alertService.success('Dosya başarılı bir şekilde eklendi.');
    if (!this.fileFormGroup.controls.file.value.name.includes('.csv') && !this.fileFormGroup.controls.file.value.name.includes('.xlsx')) {
      this.onRemove();
      this.alertService.error('Sadece "csv" veya "xlsx" uzantılı dosyalar ekleyebilirsiniz.');
    }
  }

  onRemove(): void {
    this.fileFormGroup.controls.file.setValue('');
  }

  editDataset(): void {
    this.upload(this.selection.selected[0].id);
  }

  upload(dataset_id: number = 0): void {
    this.parseFile(dataset_id);
    this.modalService.dismissAll();
  }

  parseFile(dataset_id: number): void {
    // if (this.fileFormGroup.controls.file.value.name.includes('.csv')) {
    this.parseCsv(dataset_id);
    // } else if (this.fileFormGroup.controls.file.value.name.includes('.xlsx')) {
    // this.parseXlsx();
    // }
  }

  parseCsv(dataset_id: number): void {
    if (dataset_id == 0) {
      this.papa.parse(this.fileFormGroup.controls.file.value, {
        complete: results => {
          this.rowsData = results;
          for (const [key, val] of Object.entries(results.data[0])) {
            const col = results.data.map((el: { [x: string]: any; }) => el[key]).filter((el: null) => el != null);
            if (typeof val === 'number') {
              this.numberAttributesData.push({
                distinct: new Set(col).size,
                max: Math.max.apply(Math, col),
                mean: col.reduce((a: any, b: any) => a + b, 0) / col.length,
                min: Math.min.apply(Math, col),
                missing: results.data.length - col.length,
                name: key
              });
            } else if (typeof val === 'string' || typeof val === 'boolean') {
              const tempCategories: AttributeCategory[] = [];
              const distinct = new Set<string>(col);
              distinct.forEach(value => {
                let count = 0;
                for (const el of col) {
                  if (el === value) {
                    count++;
                  }
                }
                tempCategories.push({name: value, count});
              });
              this.categoricalAttributesData.push({
                categories: tempCategories,
                distinct: distinct.size,
                missing: results.data.length - col.length,
                name: key
              });
            }
          }
          this.sendDataset.setDatasetDetail({
            id: dataset_id,
            name: this.fileFormGroup.controls.datasetName.value,
            rowsData: this.rowsData,
            categoricalAttributes: this.categoricalAttributesData,
            numberAttributes: this.numberAttributesData
          });
          this.router.navigate(['/dashboard/edit-dataset']);
        },
        dynamicTyping: true,
        skipEmptyLines: true,
        header: true,
      });
    } else {
      this.papa.parse(this.fileService.getFileUrl(this.selection.selected[0].data), {
        download: true,
        complete: results => {
          this.rowsData = results;
          for (const [key, val] of Object.entries(results.data[0])) {
            const col = results.data.map((el: { [x: string]: any; }) => el[key]).filter((el: null) => el != null);
            if (typeof val === 'number') {
              this.numberAttributesData.push({
                distinct: new Set(col).size,
                max: Math.max.apply(Math, col),
                mean: col.reduce((a: any, b: any) => a + b, 0) / col.length,
                min: Math.min.apply(Math, col),
                missing: results.data.length - col.length,
                name: key
              });
            } else if (typeof val === 'string' || typeof val === 'boolean') {
              const tempCategories: AttributeCategory[] = [];
              const distinct = new Set<string>(col);
              distinct.forEach(value => {
                let count = 0;
                for (const el of col) {
                  if (el === value) {
                    count++;
                  }
                }
                tempCategories.push({name: value, count});
              });
              this.categoricalAttributesData.push({
                categories: tempCategories,
                distinct: distinct.size,
                missing: results.data.length - col.length,
                name: key
              });
            }
          }
          this.sendDataset.setDatasetDetail({
            id: dataset_id,
            name: this.selection.selected[0].name,
            rowsData: this.rowsData,
            categoricalAttributes: this.categoricalAttributesData,
            numberAttributes: this.numberAttributesData
          });
          this.router.navigate(['/dashboard/edit-dataset']);
        },
        dynamicTyping: true,
        skipEmptyLines: true,
        header: true,
      });
    }
  }
}
