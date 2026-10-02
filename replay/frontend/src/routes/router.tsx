import { createBrowserRouter } from 'react-router-dom'
import { NotFoundPage } from '../pages/NotFoundPage'
import { ReplayPage } from '../pages/ReplayPage'

export const router = createBrowserRouter([
  { path: '/', element: <ReplayPage /> },
  { path: '/replay', element: <ReplayPage /> },
  { path: '*', element: <NotFoundPage /> },
])
