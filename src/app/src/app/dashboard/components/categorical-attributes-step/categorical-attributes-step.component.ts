import {Component, Input, OnInit, TemplateRef} from '@angular/core';
import {MatTableDataSource} from "@angular/material/table";
import {AttributeCategory, CategoricalAttribute, NumberAttribute, SendDatasetService} from "../../../core";
import {animate, state, style, transition, trigger} from "@angular/animations";
import {NgbModal} from "@ng-bootstrap/ng-bootstrap";
import {SelectionModel} from "@angular/cdk/collections";

@Component({
  selector: 'app-categorical-attributes-step',
  templateUrl: './categorical-attributes-step.component.html',
  styleUrls: ['./categorical-attributes-step.component.css'],
  animations: [
    trigger('detailExpand', [
      state('collapsed', style({height: '0px', minHeight: '0'})),
      state('expanded', style({height: '*'})),
      transition('expanded <=> collapsed', animate('225ms cubic-bezier(0.4, 0.0, 0.2, 1)')),
    ]),
  ]
})
export class CategoricalAttributesStepComponent implements OnInit {

  selection = new SelectionModel<any>(false, []);
  categoricalAttributesDataSource: MatTableDataSource<CategoricalAttribute> = new MatTableDataSource<CategoricalAttribute>();
  expandedElement?: CategoricalAttribute;

  //@ts-ignore
  last = {};

  keys = ['name', 'distinct', 'missing'];
  turkish = ['İsim', 'Ayrık', 'Eksik'];

  constructor(private sendDataset: SendDatasetService, private modalService: NgbModal) {
  }

  ngOnInit(): void {
    this.categoricalAttributesDataSource = new MatTableDataSource<CategoricalAttribute>(this.sendDataset.datasetDetail.categoricalAttributes);
  }

  getObjectKeys(object: Object) {
    return Object.keys(object);
  }

  getDataSource(data: AttributeCategory[]): MatTableDataSource<AttributeCategory> {
    return new MatTableDataSource<AttributeCategory>(data);
  }

  checkboxLabel(row?: any): string {
    if (row)
      return `${this.selection.isSelected(row) ? 'deselect' : 'select'} row ${row.position + 1}`;
    return '';
  }

  undo() {
    this.sendDataset.datasetDetail = JSON.parse(JSON.stringify(this.last));
    this.categoricalAttributesDataSource = new MatTableDataSource<NumberAttribute>(this.sendDataset.datasetDetail.categoricalAttributes);
  }

  labelEncode() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    //@ts-ignore
    let category = this.sendDataset.datasetDetail.categoricalAttributes.filter(category => {
      return category.name == this.selection.selected[0].name;
    })[0];

    let labels = {};
    // @ts-ignore
    category.categories.forEach((cat, index) => {
      // @ts-ignore
      labels[cat.name] = index;
    });
    let col = this.sendDataset.datasetDetail.rowsData.data.map(function (value: CategoricalAttribute, index: any) {
      // @ts-ignore
      value[category.name] = labels[value[category.name]];
    });
    // @ts-ignore
    this.sendDataset.datasetDetail.categoricalAttributes?.filter(cat => {
      return cat.name == category.name;
    })[0].categories.map(cat => {
      // @ts-ignore
      cat.name = labels[cat.name];
    });
    this.categoricalAttributesDataSource._updateChangeSubscription();
  }

  frequencyEncode() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    //@ts-ignore
    let category = this.sendDataset.datasetDetail.categoricalAttributes.filter(category => {
      return category.name == this.selection.selected[0].name;
    })[0];

    let labels = {};
    let total = 0;
    // @ts-ignore
    category.categories.map(cat => {
      // @ts-ignore
      total += cat.count;
    });
    // @ts-ignore
    category.categories.forEach((cat, index) => {
      // @ts-ignore
      labels[cat.name] = cat.count / total;
    });
    let col = this.sendDataset.datasetDetail.rowsData.data.map(function (value: CategoricalAttribute, index: any) {
      // @ts-ignore
      value[category.name] = labels[value[category.name]];
    });
    // @ts-ignore
    this.sendDataset.datasetDetail.categoricalAttributes?.filter(cat => {
      return cat.name == category.name;
    })[0].categories.map(cat => {
      // @ts-ignore
      cat.name = labels[cat.name];
    });
    this.categoricalAttributesDataSource._updateChangeSubscription();
  }

  oneHotEncode() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    //@ts-ignore
    let category = this.sendDataset.datasetDetail.categoricalAttributes.filter(category => {
      return category.name == this.selection.selected[0].name;
    })[0];

    let labels: {} = {};
    let counts: {} = {};
    // @ts-ignore
    category.categories.forEach((cat, index) => {
      // @ts-ignore
      labels[category.name + '_' + cat.name] = 0;
      // @ts-ignore
      counts[category.name + '_' + cat.name] = 0;
    });
    let col = this.sendDataset.datasetDetail.rowsData.data.forEach((row: any) => {
      let tmp = {...labels};
      // @ts-ignore
      tmp[category.name + '_' + row[category.name]] = 1;
      // @ts-ignore
      counts[category.name + '_' + row[category.name]] += 1;
      for (const [key, value] of Object.entries(tmp)) {
        row[key] = value;
      }
      // @ts-ignore
      delete row[category.name];
    });

    this.selection.clear();

    for (const [key, value] of Object.entries(labels)) {
      // @ts-ignore
      this.sendDataset.datasetDetail.categoricalAttributes?.push({
        name: key,
        distinct: 2,
        missing: 0,
        // @ts-ignore
        categories: [{name: '0', count: this.sendDataset.datasetDetail.rowsData.data.length - counts[key]}, {
          name: '1',
          // @ts-ignore
          count: counts[key]
        }]
      })
    }
    // @ts-ignore
    let index = this.categoricalAttributesDataSource.data.indexOf(category);
    this.categoricalAttributesDataSource.data.splice(index, 1);
    this.categoricalAttributesDataSource._updateChangeSubscription();
  }

  binaryEncode() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    //@ts-ignore
    let category = this.sendDataset.datasetDetail.categoricalAttributes.filter(category => {
      return category.name == this.selection.selected[0].name;
    })[0];

    let labels = {};
    let counts = {};
    // @ts-ignore
    category.categories.forEach((cat, index) => {
      // @ts-ignore
      labels[cat.name] = this.zeroFill((index >>> 0).toString(2), ((category.categories.length - 1) >>> 0).toString(2).length);
    });

    let col = this.sendDataset.datasetDetail.rowsData.data.forEach((row: any) => {
      // @ts-ignore
      for (const x of Array(((category.categories.length - 1) >>> 0).toString(2).length).keys()) {
        // @ts-ignore
        row[category.name + '_' + x] = labels[row[category.name]][x];
        // @ts-ignore
        if (isNaN(counts[category.name + '_' + x])) {
          // @ts-ignore
          counts[category.name + '_' + x] = 0;
        }
        // @ts-ignore
        if (labels[row[category.name]][x] == '1') {
          // @ts-ignore
          counts[category.name + '_' + x] += 1;
        }
      }
      // @ts-ignore
      delete row[category.name];
    });

    this.selection.clear();

    for (const [key, value] of Object.entries(counts)) {
      // @ts-ignore
      this.sendDataset.datasetDetail.categoricalAttributes?.push({
        name: key,
        distinct: 2,
        missing: 0,
        // @ts-ignore
        categories: [{name: '0', count: this.sendDataset.datasetDetail.rowsData.data.length - counts[key]}, {
          name: '1',
          // @ts-ignore
          count: counts[key]
        }]
      })
    }

    let index = this.categoricalAttributesDataSource.data.indexOf(category);
    this.categoricalAttributesDataSource.data.splice(index, 1);
    this.categoricalAttributesDataSource._updateChangeSubscription();
  }

  fillMax() {
    this.last = JSON.parse(JSON.stringify(this.sendDataset.datasetDetail));
    let tmpMax = '';
    let max = 0;
    this.selection.selected[0].categories.forEach((el: { count: number; name: string; }) => {
      if (el.count > max) {
        max = el.count;
        tmpMax = el.name;
      }
    });
    this.selection.selected[0].categories.forEach((el: { count: number; name: string; }) => {
      if (el.name == tmpMax) {
        el.count += this.selection.selected[0].missing;
      }
    });
    this.sendDataset.datasetDetail.rowsData.data.forEach((el: { [s: string]: unknown; } | ArrayLike<unknown>) => {
      // @ts-ignore
      if (el[this.selection.selected[0].name] == null) {
        // @ts-ignore
        el[this.selection.selected[0].name] = tmpMax;
      }
    });
    const col = this.sendDataset.datasetDetail.rowsData.data.map((el: { [x: string]: any; }) => el[this.selection.selected[0].name]).filter((el: null) => el != null);
    this.sendDataset.datasetDetail.categoricalAttributes?.map(categoricAttribute => {
      if (categoricAttribute.name == this.selection.selected[0].name) {
        categoricAttribute.distinct = new Set(col).size;
        categoricAttribute.missing = this.sendDataset.datasetDetail.rowsData.data.length - col.length;
      }
    });
    this.categoricalAttributesDataSource._updateChangeSubscription();

  }

  zeroFill(number: any, width: any) {
    width -= number.toString().length;
    if (width > 0) {
      return new Array(width + (/\./.test(number) ? 2 : 1)).join('0') + number;
    }
    return number + "";
  }
}
