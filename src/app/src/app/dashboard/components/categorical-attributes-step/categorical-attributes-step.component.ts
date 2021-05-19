import {Component, Input, OnInit} from '@angular/core';
import {MatTableDataSource} from "@angular/material/table";
import {AttributeCategory, CategoricalAttribute, NumberAttribute, SendDatasetService} from "../../../core";
import {animate, state, style, transition, trigger} from "@angular/animations";

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


  categoricalAttributesDataSource: MatTableDataSource<CategoricalAttribute> = new MatTableDataSource<CategoricalAttribute>();
  expandedElement?: CategoricalAttribute;

  keys = ['name', 'distinct', 'missing'];
  turkish = ['İsim', 'Ayrık', 'Eksik'];

  constructor(private sendDataset: SendDatasetService) {
  }

  ngOnInit(): void {
    console.log(this.sendDataset.datasetDetail);
    this.categoricalAttributesDataSource = new MatTableDataSource<CategoricalAttribute>(this.sendDataset.datasetDetail.categoricalAttributes);
  }

  getObjectKeys(object: Object) {
    return Object.keys(object);
  }

  getDataSource(data: AttributeCategory[]): MatTableDataSource<AttributeCategory> {
    return new MatTableDataSource<AttributeCategory>(data);
  }
}
