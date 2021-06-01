import {Component, OnInit} from '@angular/core';
import {ActivatedRoute} from "@angular/router";
import {ResultService} from "../../../core/services/result.service";
import {Result} from "../../../core/models/result";

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit {
  constructor(private route: ActivatedRoute,
              private resultService: ResultService) {
  }

  result?: Result;
  data?: any;
  Object = Object;

  ngOnInit(): void {
    this.resultService.getResult(this.route.snapshot.paramMap.get("id")).subscribe(res => {
      this.result = res[0];
      this.data = JSON.parse(this.result.data);
      // console.log(this.data)
    });
  }

}
