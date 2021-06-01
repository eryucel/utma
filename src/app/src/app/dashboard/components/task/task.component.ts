import {AfterViewInit, Component, OnInit, ViewChild} from '@angular/core';
import {SelectionModel} from "@angular/cdk/collections";
import {MatTableDataSource} from "@angular/material/table";
import {MatPaginator} from "@angular/material/paginator";
import {MatSort} from "@angular/material/sort";
import {NgbModal} from "@ng-bootstrap/ng-bootstrap";
import {FormBuilder, FormGroup, Validators} from "@angular/forms";
import {TaskService} from "../../../core/services/task.service";
import {DatasetService} from "../../../core/services/dataset.service";
import {Dataset} from "../../../core/models/dataset";
import {Task} from "../../../core/models/task";
import {interval} from 'rxjs';
import {Router} from "@angular/router";

@Component({
  selector: 'app-task',
  templateUrl: './task.component.html',
  styleUrls: ['./task.component.css']
})
export class TaskComponent implements OnInit, AfterViewInit {
  selection = new SelectionModel<any>(false, []);
  displayedColumns: string[] = ['select', 'dataset', 'algorithm', 'status', 'completed_date', 'create_date'];
  dataSource: MatTableDataSource<any>;

  fileFormGroup: FormGroup;
  datasets: Dataset[] = [];

  @ViewChild(MatPaginator) paginator: MatPaginator | null = null;
  @ViewChild(MatSort) sort: MatSort | null = null;

  algorithmType = 0;
  algorithm = 0;
  dataset ?: Dataset;
  test_split = false;
  percentage = 20;
  n_cluster = 4;
  predicted_column = "";
  test_dataset ?: Dataset;

  constructor(
    private modalService: NgbModal,
    private formBuilder: FormBuilder,
    private taskService: TaskService, private router: Router,
    private datasetService: DatasetService) {
    this.fileFormGroup = this.formBuilder.group({
      datasetName: ['', Validators.required],
      file: ['', Validators.required]
    });
    this.dataSource = new MatTableDataSource();
  }

  ngOnInit(): void {
    this.datasetService.getDatasets().subscribe(result => {
      this.datasets = result;
    });

    this.taskService.getTasks().subscribe((results: Task[] | undefined) => {
      this.dataSource = new MatTableDataSource(results);
      this.dataSource.paginator = this.paginator;
      this.dataSource.sort = this.sort;
    });

    this.taskService.getTasksInterval().subscribe((results: Task[] | undefined) => {
      this.dataSource = new MatTableDataSource(results);
      this.dataSource.paginator = this.paginator;
      this.dataSource.sort = this.sort;
    });
  }

  ngAfterViewInit(): void {
    this.dataSource.paginator = this.paginator;
    this.dataSource.sort = this.sort;
  }

  resetAll() {
    this.algorithmType = 0;
    this.algorithm = 0;
    this.dataset = undefined;
    this.test_split = false;
    this.percentage = 20;
    this.n_cluster = 4;
    this.predicted_column = "";
    this.test_dataset = undefined;
  }

  applyFilter(event: Event): void {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();

    if (this.dataSource.paginator) {
      this.dataSource.paginator.firstPage();
    }
  }

  getAttrs(dataset: Dataset) {
    let attrs = dataset.numericAttributes.split(',');
    return attrs.concat(dataset.categoricalAttributes.split(','))
  }

  checkboxLabel(row?: any): string {
    return `${this.selection.isSelected(row) ? 'deselect' : 'select'} row ${row.position + 1}`;
  }

  open(content: any) {
    this.modalService.open(content, {ariaLabelledBy: 'modal-basic-title'});
  }

  run() {
    const algorithmNames = ['Adaboost Classification', 'Decision Tree Classification', 'Gaussian Naive Bayes Classification', 'Gradient Boosting Classification', 'KNN Classification', 'Logistic Regression Classification', 'Random Forest Classification', 'SVM Classification', 'Adaboost Regression', 'Bayesian Ridge Regression', 'Gradient Boosting Regression', 'Lasso Regression', 'Linear Regression', 'Random Forest Regression', 'SGD Regression', 'DBSCAN', 'Hierarchical Clustering', 'K-means', 'Classification Model Optimize', 'Regression Model Optimize'];
    let parameters = {};
    // @ts-ignore
    parameters.dataset = this.dataset?.data;
    // @ts-ignore
    parameters.categorical_columns = this.dataset?.categoricalAttributes;
    if (this.algorithm < 16 || this.algorithm > 18) {
      // @ts-ignore
      parameters.predicted_column = this.predicted_column;
      // @ts-ignore
      parameters.train_test_split = this.test_split;
      // @ts-ignore
      parameters.supplied_test_set = this.test_dataset;
      // @ts-ignore
      parameters.percentage_split = this.percentage / 100;
    } else if (this.algorithm == 17) {
      // @ts-ignore
      parameters.n_cluster = this.n_cluster;
    }
    this.taskService.postTask({
      id: undefined,
      dataset: this.dataset?.id,
      datasetName: this.dataset?.name,
      algorithm: this.algorithm,
      algorithmName: algorithmNames[this.algorithm - 1],
      status: 0,
      completed_date: undefined,
      create_date: undefined,
      parameters: JSON.stringify(parameters)
    }).subscribe(result => {
      this.dataSource.data.push(result);
      this.dataSource._updateChangeSubscription();
      this.taskService.runTask(result.id);
    });
    this.modalService.dismissAll();
    this.resetAll();
  }

  view() {
    window.open('http://localhost:4200/dashboard/result/' + this.selection.selected[0].id, '_blank')
  }

}
