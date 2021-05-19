import {Component, Input, OnInit} from '@angular/core';
import {MatTableDataSource} from "@angular/material/table";
import {NumberAttribute, SendDatasetService} from "../../../core";

@Component({
  selector: 'app-number-attributes-step',
  templateUrl: './number-attributes-step.component.html',
  styleUrls: ['./number-attributes-step.component.css']
})
export class NumberAttributesStepComponent implements OnInit {

  numberAttributesDataSource: MatTableDataSource<NumberAttribute> = new MatTableDataSource<NumberAttribute>();

  constructor(private sendDataset: SendDatasetService) {
  }

  ngOnInit(): void {
    this.numberAttributesDataSource = new MatTableDataSource<NumberAttribute>(this.sendDataset.datasetDetail.numberAttributes);
  }

}
