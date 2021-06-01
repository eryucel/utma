import {Component, Input, OnInit} from '@angular/core';
import {MatTableDataSource} from "@angular/material/table";
import {AttributeCategory, NumberAttribute, SendDatasetService} from "../../../core";
import {SelectionModel} from "@angular/cdk/collections";

@Component({
  selector: 'app-number-attributes-step',
  templateUrl: './number-attributes-step.component.html',
  styleUrls: ['./number-attributes-step.component.css']
})
export class NumberAttributesStepComponent implements OnInit {

  selection = new SelectionModel<any>(false, []);
  numberAttributesDataSource: MatTableDataSource<NumberAttribute> = new MatTableDataSource<NumberAttribute>();

  //@ts-ignore
  last = {};

  constructor(private sendDataset: SendDatasetService) {
  }

  ngOnInit(): void {
    this.numberAttributesDataSource = new MatTableDataSource<NumberAttribute>(this.sendDataset.datasetDetail.numberAttributes);
  }


  checkboxLabel(row?: any): string {
    return `${this.selection.isSelected(row) ? 'deselect' : 'select'} row ${row.position + 1}`;
  }

  undo() {
    this.sendDataset.datasetDetail = JSON.parse(JSON.stringify(this.last));
    this.numberAttributesDataSource = new MatTableDataSource<NumberAttribute>(this.sendDataset.datasetDetail.numberAttributes);
  }

  randomFill() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    this.sendDataset.datasetDetail.rowsData.data.forEach((el: { [s: string]: unknown; } | ArrayLike<unknown>) => {
      // @ts-ignore
      if (el[this.selection.selected[0].name] == null) {
        // @ts-ignore
        el[this.selection.selected[0].name] = (Math.random() * (this.selection.selected[0].max - this.selection.selected[0].min) + this.selection.selected[0].min);
      }
    });
    const col = this.sendDataset.datasetDetail.rowsData.data.map((el: { [x: string]: any; }) => el[this.selection.selected[0].name]).filter((el: null) => el != null);
    this.sendDataset.datasetDetail.numberAttributes?.map(numberAttribute => {
      if (numberAttribute.name == this.selection.selected[0].name) {
        numberAttribute.distinct = new Set(col).size;
        numberAttribute.max = Math.max.apply(Math, col);
        numberAttribute.mean = col.reduce((a: any, b: any) => a + b, 0) / col.length;
        numberAttribute.min = Math.min.apply(Math, col);
        numberAttribute.missing = this.sendDataset.datasetDetail.rowsData.data.length - col.length;
      }
    });
    this.numberAttributesDataSource._updateChangeSubscription();
  }

  meanFill() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    this.sendDataset.datasetDetail.rowsData.data.forEach((el: { [s: string]: unknown; } | ArrayLike<unknown>) => {
      // @ts-ignore
      if (el[this.selection.selected[0].name] == null) {
        // @ts-ignore
        el[this.selection.selected[0].name] = this.selection.selected[0].mean;
      }
    });
    const col = this.sendDataset.datasetDetail.rowsData.data.map((el: { [x: string]: any; }) => el[this.selection.selected[0].name]).filter((el: null) => el != null);
    this.sendDataset.datasetDetail.numberAttributes?.map(numberAttribute => {
      if (numberAttribute.name == this.selection.selected[0].name) {
        numberAttribute.distinct = new Set(col).size;
        numberAttribute.max = Math.max.apply(Math, col);
        numberAttribute.mean = col.reduce((a: any, b: any) => a + b, 0) / col.length;
        numberAttribute.min = Math.min.apply(Math, col);
        numberAttribute.missing = this.sendDataset.datasetDetail.rowsData.data.length - col.length;
      }
    });
    this.numberAttributesDataSource._updateChangeSubscription();
  }
}
