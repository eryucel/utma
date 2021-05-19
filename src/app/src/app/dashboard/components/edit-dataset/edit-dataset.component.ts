import {Component, OnInit} from '@angular/core';
import {NumberAttribute, SendDatasetService} from "../../../core";
import {UploadedDataset} from "../../../core/models/uploadedDataset";
import {MatTableDataSource} from "@angular/material/table";

@Component({
  selector: 'app-edit-dataset',
  templateUrl: './edit-dataset.component.html',
  styleUrls: ['./edit-dataset.component.css']
})
export class EditDatasetComponent implements OnInit {

  constructor() {
  }

  ngOnInit(): void {
  }

}
