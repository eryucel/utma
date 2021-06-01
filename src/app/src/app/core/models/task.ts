export class Task {
  id: number | undefined = 0;
  algorithmName: string = ''
  algorithm: number = 0;
  datasetName?: string = '';
  dataset?: number = 0;
  status: number = 0;
  parameters: string = '';
  completed_date?: string = '';
  create_date?: string = '';
}
