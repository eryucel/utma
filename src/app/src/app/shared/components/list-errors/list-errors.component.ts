import {Component, Input, OnInit} from '@angular/core';
import {Errors} from "../../../core";

@Component({
  selector: 'app-list-errors',
  templateUrl: './list-errors.component.html',
  styleUrls: ['./list-errors.component.css']
})
export class ListErrorsComponent implements OnInit {

  formattedErrors: Array<string> = [];

  constructor() {
  }

  ngOnInit(): void {
  }


  @Input()
  set errors(errorList: Errors) {
    this.formattedErrors = Object.keys(errorList.errors || {})
      .map(key => `${key} ${errorList.errors[key]}`);
  }

  get errorList() {
    return this.formattedErrors;
  }
}
